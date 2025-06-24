import argparse
import logging
import os
import torch
from pycocotools.coco import COCO
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.dataloader import BatchImageCollateFuncion, DataLoader
from data.dataset.coco_dataset import CocoDetection
from data.dataset.coco_eval import CocoEvaluator
from data.transforms.compose import Compose
from data.transforms.transforms import RandomPhotometricDistort, RandomZoomOut, RandomIoUCrop, RandomHorizontalFlip, \
    SanitizeBoundingBoxes, Resize, ConvertPILImage, ConvertBoxes
from models.modules.criterion import RTDETRCriterionv2
from models.modules.decoder import RTDETRTransformerv2
from models.modules.ema import ModelEMA
from models.modules.encoder import HybridEncoder
from models.modules.matcher import HungarianMatcher
from models.modules.postprocessor import RTDETRPostProcessor
from models.modules.presnet import PResNet
from models.rtdetr import RTDETR
from trainer import train_one_epoch, evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from", default=None)
parser.add_argument("--epochs", type=int, default=120)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--output_dir", type=str, default="checkpoints")
parser.add_argument("--disable_wandb", action="store_true")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("RT-DETR")

use_wandb = not args.disable_wandb
if use_wandb:
    import wandb
    wandb.init(project="RT-DETR-Train", config=vars(args))

device = "cuda" if torch.cuda.is_available() else "cpu"

train_trans = Compose([
    RandomPhotometricDistort(p=0.5),
    RandomZoomOut(fill=0),
    RandomIoUCrop(p=0.8),
    SanitizeBoundingBoxes(min_size=1),
    RandomHorizontalFlip(),
    Resize(size=(640, 640)),
    SanitizeBoundingBoxes(min_size=1),
    ConvertPILImage(dtype=torch.float32, scale=True),
    ConvertBoxes(fmt='cxcywh', normalize=True)
], disable_first_n=3, stop_epoch=117)

val_trans = Compose([
    Resize(size=(640, 640)),
    ConvertPILImage(dtype=torch.float32, scale=True),
], disable_first_n=0)

train_dataset = CocoDetection("train2017", "annotations/instances_train2017.json", transforms=train_trans,
                              remap_mscoco_category=True)
val_dataset = CocoDetection("val2017", "annotations/instances_val2017.json", transforms=val_trans,
                            remap_mscoco_category=True)

train_collate = BatchImageCollateFuncion(scales=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800],
                                         stop_epoch=117)
val_collate = BatchImageCollateFuncion()

train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=train_collate)
val_dataloader = DataLoader(val_dataset, args.batch_size * 2, shuffle=False, collate_fn=val_collate)

backbone = PResNet(18, pretrained=True)
encoder = HybridEncoder()
decoder = RTDETRTransformerv2(num_classes=80)
model = RTDETR(backbone, encoder, decoder).to(device)
ema = ModelEMA(model)

matcher = HungarianMatcher({'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2}, use_focal_loss=True)
criterion = RTDETRCriterionv2(matcher, {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2}, losses=['vfl', 'boxes'], num_classes=80)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
scaler = torch.amp.GradScaler(device)
postprocessor = RTDETRPostProcessor(num_classes=80, remap_mscoco_category=True)
coco_evaluator = CocoEvaluator(COCO("annotations/instances_val2017.json"))

start_epoch = 1
best_score = 0.0

if args.resume:
    logger.info(f"Resuming from checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint["ema_state"]["module"])
    ema.load_state_dict(checkpoint["ema_state"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    best_score = checkpoint.get("best_score", 0.0)
    start_epoch = checkpoint["epoch"] + 1

for epoch in range(start_epoch, args.epochs + 1):
    train_dataloader.set_epoch(epoch)
    val_dataloader.set_epoch(epoch)

    avg_epoch_loss = train_one_epoch(model, ema, criterion, train_dataloader, optimizer, scaler, epoch, device, use_wandb)
    val_score = evaluate(model, postprocessor, coco_evaluator, val_dataloader, device, epoch, use_wandb)

    scheduler.step()

    logger.info(
        f"[Epoch {epoch}] Train Loss: {avg_epoch_loss:.4f} | Val mAP: {val_score:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

    if val_score > best_score:
        best_score = val_score
        save_path = os.path.join(args.output_dir, "model.pth")
        torch.save({
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_score": best_score
        }, save_path)
        logger.info(f"Saved checkpoint to {save_path}")

if use_wandb:
    wandb.finish()