import contextlib
import copy
import os

import numpy as np
import torch
from faster_coco_eval import COCO, COCOeval_faster


class CocoEvaluator(object):
    def __init__(self, coco_gt):
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.coco_eval = COCOeval_faster(coco_gt, iouType="bbox", print_function=print, separate_eval=True)
        self.img_ids = []
        self.eval_imgs = []

    def cleanup(self):
        self.coco_eval = COCOeval_faster(self.coco_gt, iouType="bbox", print_function=print, separate_eval=True)
        self.img_ids = []
        self.eval_imgs = []

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        results = self.prepare(predictions)

        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                coco_dt = self.coco_gt.loadRes(results) if results else COCO()
                self.coco_eval.cocoDt = coco_dt
                self.coco_eval.params.imgIds = list(img_ids)
                self.coco_eval.evaluate()

        self.eval_imgs.append(np.array(self.coco_eval._evalImgs_cpp).reshape(len(self.coco_eval.params.catIds),
                                                                             len(self.coco_eval.params.areaRng),
                                                                             len(self.coco_eval.params.imgIds)))

    def accumulate(self):
        self.coco_eval.accumulate()

    def summarize(self):
        self.coco_eval.summarize()

    def prepare(self, predictions):
        return self.prepare_for_coco_detection(predictions)

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            xmin, ymin, xmax, ymax = boxes.unbind(1)
            boxes = torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results
