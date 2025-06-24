from typing import Any, Dict, List, Optional

import torch
import torchvision
import torchvision.transforms.v2 as T

from ..utils import _boxes_keys
from ..utils import convert_to_tv_tensor

RandomPhotometricDistort = T.RandomPhotometricDistort
RandomZoomOut = T.RandomZoomOut
SanitizeBoundingBoxes = T.SanitizeBoundingBoxes
RandomHorizontalFlip = T.RandomHorizontalFlip
Resize = T.Resize
RandomCrop = T.RandomCrop
Normalize = T.Normalize


class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5,
                 max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40,
                 p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


class ConvertBoxes:
    def __init__(self, fmt='', normalize=False):
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def __call__(self, inpt: Any, params: Dict[str, Any], dataset) -> Any:
        spatial_size = getattr(params['boxes'], _boxes_keys[1])
        if self.fmt:
            in_fmt = params['boxes'].format.value.lower()
            params['boxes'] = torchvision.ops.box_convert(params['boxes'], in_fmt=in_fmt, out_fmt=self.fmt.lower())
            params['boxes'] = convert_to_tv_tensor(params['boxes'], key='boxes', box_format=self.fmt.upper(),
                                                   spatial_size=spatial_size)

        if self.normalize:
            params['boxes'] = params['boxes'] / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt, params, dataset


class ConvertPILImage:
    def __init__(self, dtype='float32', scale=True):
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def __call__(self, inpt: Any, params: Dict[str, Any], dataset) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = torchvision.tv_tensors.Image(inpt)
        return inpt, params, dataset
