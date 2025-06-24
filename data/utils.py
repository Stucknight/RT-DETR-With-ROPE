from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Mask

_boxes_keys = ['format', 'canvas_size']


def convert_to_tv_tensor(tensor, key: str, box_format='xyxy', spatial_size=None):
    if key == 'boxes':
        box_format = getattr(BoundingBoxFormat, box_format.upper())
        _kwargs = dict(zip(_boxes_keys, [box_format, spatial_size]))
        return BoundingBoxes(tensor, **_kwargs)
    if key == 'masks':
        return Mask(tensor)
    return None