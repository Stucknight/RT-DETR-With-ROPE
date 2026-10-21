import argparse
import logging

import torch
from pycocotools.coco import COCO

from data.dataloader import BatchImageCollateFuncion, DataLoader
from data.dataset.coco_dataset import CocoDetection
from data.dataset.coco_eval import CocoEvaluator
from data.transforms.compose import Compose
from data.transforms.transforms import Resize, ConvertPILImage
from models.modules.decoder import RTDETRTransformerv2
from models.modules.ema import ModelEMA
from models.modules.encoder import HybridEncoder
from models.modules.postprocessor import RTDETRPostProcessor
from models.modules.presnet import PResNet
from models.rtdetr import RTDETR
from trainer import evaluate


def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger("RT-DETR-Eval")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_trans = Compose([
        Resize(size=(640, 640)),
        ConvertPILImage(dtype=torch.float32, scale=True),
    ], disable_first_n=0)

    val_dataset = CocoDetection("val2017", "annotations/instances_val2017.json", transforms=val_trans,
                                remap_mscoco_category=True)
    val_collate = BatchImageCollateFuncion()
    val_dataloader = DataLoader(val_dataset, args.batch_size, shuffle=False, collate_fn=val_collate,
                                num_workers=args.num_workers, pin_memory=True)

    backbone = PResNet(18)
    encoder = HybridEncoder()
    decoder = RTDETRTransformerv2(num_classes=80)
    model = RTDETR(backbone, encoder, decoder).to(device)
    ema = ModelEMA(model)

    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    ema.load_state_dict(checkpoint["ema"])
    model.load_state_dict(checkpoint["ema"]["module"])

    postprocessor = RTDETRPostProcessor(num_classes=80, remap_mscoco_category=True)
    coco_evaluator = CocoEvaluator(COCO("annotations/instances_val2017.json"))

    val_score = evaluate(model, postprocessor, coco_evaluator, val_dataloader, device, epoch=None, use_wandb=False)
    logger.info(f"Validation mAP: {val_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RT-DETR on COCO validation set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()
    main(args)
