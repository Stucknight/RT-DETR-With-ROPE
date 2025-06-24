import torchvision.transforms.v2 as T


class Compose(T.Compose):
    def __init__(self, transforms, disable_first_n=0, stop_epoch=10):
        super().__init__(transforms)
        self._disable_first_n = disable_first_n
        self._stop_epoch = stop_epoch
        self._epoch = 0

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __call__(self, img, target=None, context=None):
        if self._epoch < self._stop_epoch:
            transforms_to_use = self.transforms
        else:
            transforms_to_use = self.transforms[self._disable_first_n:]

        for t in transforms_to_use:
            if target is not None and context is not None:
                img, target, context = t(img, target, context)
            elif target is not None:
                img, target = t(img, target)
            else:
                img = t(img)
        if target is not None and context is not None:
            return img, target, context
        elif target is not None:
            return img, target
        else:
            return img