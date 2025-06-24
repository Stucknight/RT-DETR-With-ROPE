import torch.nn as nn


class RTDETR(nn.Module):
    def __init__(self, backbone, encoder, decoder):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)
        return x
