import torch
from tqdm import tqdm


def train_one_epoch(model, ema, criterion, dataloader, optimizer, scaler, epoch, device, use_wandb=False):
    model.train()
    criterion.train()
    dataloader.set_epoch(epoch)

    epoch_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for i, (samples, targets) in enumerate(pbar):
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        with torch.autocast(device_type='cuda', cache_enabled=True):
            outputs = model(samples, targets=targets)

        with torch.autocast(device_type='cuda', enabled=False):
            loss_dict = criterion(outputs, targets, epoch=epoch, step=i)

        loss = sum(loss_dict.values())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        ema.update(model)

        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

        if use_wandb:
            import wandb
            wandb.log({
                "loss_vfl": loss_dict.get("loss_vfl", 0.0),
                "loss_giou": loss_dict.get("loss_giou", 0.0),
                "loss_bbox": loss_dict.get("loss_bbox", 0.0),
                "step_loss": loss.item(),
                "epoch": epoch,
                "lr": optimizer.param_groups[0]['lr']
            })

    avg_loss = epoch_loss / len(dataloader)
    return avg_loss


def evaluate(model, postprocessor, coco_evaluator, dataloader, device, epoch, use_wandb=False):
    model.eval()
    coco_evaluator.cleanup()

    with torch.no_grad():
        for samples, targets in tqdm(dataloader, desc="Validation"):
            samples = samples.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            outputs = model(samples)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessor(outputs, orig_target_sizes)
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            coco_evaluator.update(res)
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        stats = coco_evaluator.coco_eval.stats.tolist()

        if use_wandb:
            import wandb
            wandb.log({
                "mAP@[0.5:0.95]": stats[0],
                "AP@0.5": stats[1],
                "AP@0.75": stats[2],
                "AR@1": stats[6],
                "AR@10": stats[7],
                "AR@100": stats[8],
            })

        return stats[0]
