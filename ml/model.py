from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleDS(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(DSConv(in_ch, out_ch), DSConv(out_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyGuideUNet(nn.Module):
    """Small U-Net for 512x288 native-guide segmentation."""

    def __init__(self, num_classes: int = 3, base: int = 16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.SiLU(inplace=True),
        )
        self.e1 = DoubleDS(base, base)
        self.e2 = DoubleDS(base, base * 2)
        self.e3 = DoubleDS(base * 2, base * 4)
        self.e4 = DoubleDS(base * 4, base * 6)
        self.bottleneck = DoubleDS(base * 6, base * 8)

        self.d4 = DoubleDS(base * 8 + base * 6, base * 6)
        self.d3 = DoubleDS(base * 6 + base * 4, base * 4)
        self.d2 = DoubleDS(base * 4 + base * 2, base * 2)
        self.d1 = DoubleDS(base * 2 + base, base)
        self.head = nn.Conv2d(base, num_classes, 1)

    @staticmethod
    def _down(x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, 2)

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x1 = self.e1(x)
        x2 = self.e2(self._down(x1))
        x3 = self.e3(self._down(x2))
        x4 = self.e4(self._down(x3))
        xb = self.bottleneck(self._down(x4))

        y4 = self.d4(torch.cat([self._up(xb, x4), x4], dim=1))
        y3 = self.d3(torch.cat([self._up(y4, x3), x3], dim=1))
        y2 = self.d2(torch.cat([self._up(y3, x2), x2], dim=1))
        y1 = self.d1(torch.cat([self._up(y2, x1), x1], dim=1))
        return self.head(y1)


def build_model(num_classes: int = 3) -> TinyGuideUNet:
    return TinyGuideUNet(num_classes=num_classes)
