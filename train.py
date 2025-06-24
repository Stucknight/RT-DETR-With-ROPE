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

train_dataloader = DataLoader(train_dataset, 16, shuffle=True, collate_fn=train_collate)
val_dataloader = DataLoader(val_dataset, 32, shuffle=False, collate_fn=val_collate)

device = "cuda" if torch.cuda.is_available() else "cpu"
backbone = PResNet(18, pretrained=True)
encoder = HybridEncoder()
decoder = RTDETRTransformerv2(num_classes=80)
model = RTDETR(backbone, encoder, decoder).to(device)

ema = ModelEMA(model)

coco_evaluator = CocoEvaluator(COCO("annotations/instances_val2017.json"))
postprocessor = RTDETRPostProcessor(num_classes=80, remap_mscoco_category=True)
matcher = HungarianMatcher({'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2}, use_focal_loss=True)
criterion = RTDETRCriterionv2(matcher, {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2}, losses=['vfl', 'boxes'],
                              num_classes=80)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=120)
scaler = torch.amp.GradScaler(device)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("RT-DETR")

use_wandb = True

if use_wandb:
    import wandb

    wandb.init(project="RT-DETR-Train")

best_score = 0.0

for epoch in range(1, 121):
    train_dataloader.set_epoch(epoch)
    val_dataloader.set_epoch(epoch)

    avg_epoch_loss = train_one_epoch(
        model, ema, criterion, train_dataloader, optimizer, scaler, epoch, device, use_wandb
    )

    val_score = evaluate(
        model, postprocessor, coco_evaluator, val_dataloader, device, epoch, use_wandb
    )

    scheduler.step()

    logger.info(
        f"[Epoch {epoch}] Train Loss: {avg_epoch_loss:.4f} | Val mAP: {val_score:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    if val_score > best_score:
        best_score = val_score
        save_path = os.path.join("checkpoints", "model.pth")
        torch.save(ema.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")

if use_wandb:
    wandb.finish()
