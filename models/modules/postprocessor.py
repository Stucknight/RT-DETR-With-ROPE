import torch
import torch.nn as nn
import torchvision

from ...data.dataset.coco_dataset import mscoco_label2category

def mod(a, b):
    out = a - a // b * b
    return out


class RTDETRPostProcessor(nn.Module):
    def __init__(
            self,
            num_classes=80,
            use_focal_loss=True,
            num_top_queries=300,
            remap_mscoco_category=False
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = nn.functional.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

        else:
            scores = nn.functional.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        if self.remap_mscoco_category:
            labels = (torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])
                      .to(boxes.device).reshape(labels.shape))
        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        return results