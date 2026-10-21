import argparse
import logging
import math
import os
import re

import torch
from pycocotools.coco import COCO
from torch.optim.lr_scheduler import LambdaLR

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


def load_optimizer_state(optimizer, state, model):
    if len(state["param_groups"]) == len(optimizer.param_groups):
        optimizer.load_state_dict(state)
        return
    # Older checkpoints used a single param group over model.parameters(); carry each parameter's state over
    old_index = {id(p): i for i, p in enumerate(model.parameters())}
    params = [p for group in optimizer.param_groups for p in group["params"]]
    new_state = optimizer.state_dict()
    new_state["state"] = {j: state["state"][old_index[id(p)]] for j, p in enumerate(params)
                          if old_index[id(p)] in state["state"]}
    optimizer.load_state_dict(new_state)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger("RT-DETR")

    use_wandb = not args.disable_wandb
    if use_wandb:
        import wandb
        wandb.init(project="RT-DETR-Train", config=vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

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

    # Workers are re-created every epoch (no persistent_workers) so they pick up set_epoch() for the
    # augmentation / multi-scale schedules.
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=train_collate,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, args.batch_size * 2, shuffle=False, collate_fn=val_collate,
                                num_workers=args.num_workers, pin_memory=True)

    backbone = PResNet(18, pretrained=True)
    encoder = HybridEncoder()
    decoder = RTDETRTransformerv2(num_classes=80)
    model = RTDETR(backbone, encoder, decoder).to(device)
    ema = ModelEMA(model)

    matcher = HungarianMatcher({'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2}, use_focal_loss=True)
    criterion = RTDETRCriterionv2(matcher, {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2}, losses=['vfl', 'boxes'],
                                  alpha=0.75, gamma=2.0, num_classes=80)

    # No weight decay on BatchNorm / LayerNorm parameters
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    param_groups = [
        {"params": [p for n, p in params if not re.search(r"(?:norm|bn)", n)]},
        {"params": [p for n, p in params if re.search(r"(?:norm|bn)", n)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay,
                                  fused=device == "cuda")

    # Stepped every iteration: linear warmup over the first warmup_iters, times a cosine decay that changes once
    # per epoch (epoch e uses 0.5 * (1 + cos(pi * (e - 1) / epochs))).
    iters_per_epoch = len(train_dataloader)

    def lr_lambda(it):
        warmup = min(1.0, (it + 1) / args.warmup_iters) if args.warmup_iters > 0 else 1.0
        return warmup * 0.5 * (1 + math.cos(math.pi * (it // iters_per_epoch) / args.epochs))

    scheduler = LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(device)
    postprocessor = RTDETRPostProcessor(num_classes=80, remap_mscoco_category=True)
    coco_evaluator = CocoEvaluator(COCO("annotations/instances_val2017.json"))

    start_epoch = 1
    best_score = 0.0

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        # Older checkpoints only stored the EMA weights
        model.load_state_dict(checkpoint.get("model", checkpoint["ema"]["module"]))
        ema.load_state_dict(checkpoint["ema"])
        load_optimizer_state(optimizer, checkpoint["optimizer"], model)
        if "scheduler" in checkpoint and "lr_lambdas" in checkpoint["scheduler"]:
            scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            # Older checkpoints used a per-epoch scheduler; jump to the first iteration of the next epoch
            scheduler.last_epoch = checkpoint["epoch"] * iters_per_epoch - 1
            scheduler.step()
        scaler.load_state_dict(checkpoint["scaler"])
        best_score = checkpoint.get("best_score", 0.0)
        start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch, args.epochs + 1):
        train_dataloader.set_epoch(epoch)
        val_dataloader.set_epoch(epoch)

        avg_epoch_loss = train_one_epoch(model, ema, criterion, train_dataloader, optimizer, scheduler, scaler, epoch,
                                         device, args.clip_max_norm, use_wandb)
        val_score = evaluate(model, postprocessor, coco_evaluator, val_dataloader, device, epoch, use_wandb)

        logger.info(
            f"[Epoch {epoch}] Train Loss: {avg_epoch_loss:.4f} | Val mAP: {val_score:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_score > best_score:
            best_score = val_score
            save_path = os.path.join(args.output_dir, "model.pth")
            torch.save({
                "model": model.state_dict(),
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


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(description="Train RT-DETR on COCO dataset")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from", default=None)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_iters", type=int, default=2000, help="Linear LR warmup iterations (0 to disable)")
    parser.add_argument("--clip_max_norm", type=float, default=0.1, help="Gradient clipping max norm (0 to disable)")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--disable_wandb", action="store_true")

    if len(sys.argv) == 1:
        parser.print_help()
        print("\nExample:")
        print("  python train.py --epochs 100 --batch_size 8 --lr 5e-5 --output_dir checkpoints")
        sys.exit(1)

    args = parser.parse_args()
    main(args)
