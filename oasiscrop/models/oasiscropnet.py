from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        if padding is None:
            padding = dilation * (kernel_size - 1) // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class ResNet50Encoder(nn.Module):
    """ResNet-50 feature extractor returning 1/4 to 1/32 scale features."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = ResNet50_Weights.IMAGENET1K_V2
            except AttributeError:
                weights = ResNet50_Weights.DEFAULT
        try:
            net = resnet50(weights=weights)
        except Exception:
            # Keep the experiment runnable on machines without cached weights or network access.
            net = resnet50(weights=None)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.out_channels = OrderedDict(
            c2=256,
            c3=512,
            c4=1024,
            c5=2048,
        )

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return OrderedDict(c2=c2, c3=c3, c4=c4, c5=c5)


class ASPP(nn.Module):
    """Atrous spatial pyramid pooling for oasis-field context."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        rates: tuple[int, ...] = (1, 6, 12, 18),
    ) -> None:
        super().__init__()
        branches: list[nn.Module] = []
        for rate in rates:
            if rate == 1:
                branches.append(ConvBNAct(in_channels, out_channels, kernel_size=1, padding=0))
            else:
                branches.append(
                    ConvBNAct(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        dilation=rate,
                        padding=rate,
                    )
                )
        self.branches = nn.ModuleList(branches)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            ConvBNAct(out_channels * (len(rates) + 1), out_channels, kernel_size=1, padding=0),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        feats = [branch(x) for branch in self.branches]
        pooled = self.image_pool(x)
        pooled = F.interpolate(pooled, size=size, mode="bilinear", align_corners=False)
        feats.append(pooled)
        return self.project(torch.cat(feats, dim=1))


class FPNDecoder(nn.Module):
    """Top-down FPN decoder producing a 1/4-scale spatial feature map."""

    def __init__(
        self,
        in_channels: tuple[int, int, int, int] = (256, 512, 1024, 2048),
        out_channels: int = 256,
    ) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(ch, out_channels, kernel_size=1, bias=False) for ch in in_channels]
        )
        self.smooth = nn.ModuleList(
            [ConvBNAct(out_channels, out_channels, kernel_size=3) for _ in in_channels]
        )

    def forward(self, features: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
        c2, c3, c4, c5 = features["c2"], features["c3"], features["c4"], features["c5"]
        feats = [c2, c3, c4, c5]
        laterals = [lat(feat) for lat, feat in zip(self.lateral, feats)]

        p5 = laterals[3]
        p4 = laterals[2] + F.interpolate(p5, size=laterals[2].shape[-2:], mode="bilinear", align_corners=False)
        p3 = laterals[1] + F.interpolate(p4, size=laterals[1].shape[-2:], mode="bilinear", align_corners=False)
        p2 = laterals[0] + F.interpolate(p3, size=laterals[0].shape[-2:], mode="bilinear", align_corners=False)

        return self.smooth[0](p2)


class BoundaryAuxiliaryHead(nn.Module):
    """Explicit boundary head for semantic cropland-edge supervision."""

    def __init__(self, in_channels: int = 256, hidden_channels: int = 96) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels, kernel_size=3),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3),
        )
        self.logits = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x)
        return feat, self.logits(feat)


class BoundaryGuidedFusion(nn.Module):
    """Fusion block combining spatial, contextual, and boundary features."""

    def __init__(
        self,
        spatial_channels: int = 256,
        context_channels: int = 256,
        boundary_channels: int = 96,
        out_channels: int = 256,
    ) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            ConvBNAct(spatial_channels + context_channels + boundary_channels, out_channels, kernel_size=1, padding=0),
            ConvBNAct(out_channels, out_channels, kernel_size=3),
            nn.Dropout2d(0.1),
        )

    def forward(
        self,
        spatial: torch.Tensor,
        context: torch.Tensor,
        boundary: torch.Tensor,
    ) -> torch.Tensor:
        context = F.interpolate(context, size=spatial.shape[-2:], mode="bilinear", align_corners=False)
        boundary = F.interpolate(boundary, size=spatial.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([spatial, context, boundary], dim=1))


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int = 256, mid_channels: int = 128, num_classes: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_channels, mid_channels, kernel_size=3),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class OasisCropNet(nn.Module):
    """Boundary-guided FPN model for OasisCrop net-cropland segmentation.

    The architecture combines multiscale contextual decoding with explicit
    boundary supervision for visible net annual-crop segmentation.
    """

    def __init__(
        self,
        num_class: int = 1,
        pretrained_backbone: bool = True,
        fpn_channels: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = ResNet50Encoder(pretrained=pretrained_backbone)
        self.fpn = FPNDecoder(out_channels=fpn_channels)
        self.context = ASPP(2048, out_channels=fpn_channels)
        self.boundary = BoundaryAuxiliaryHead(fpn_channels, hidden_channels=96)
        self.fusion = BoundaryGuidedFusion(
            spatial_channels=fpn_channels,
            context_channels=fpn_channels,
            boundary_channels=96,
            out_channels=fpn_channels,
        )
        self.main_head = SegmentationHead(fpn_channels, mid_channels=128, num_classes=num_class)
        self.aux_head = SegmentationHead(fpn_channels, mid_channels=96, num_classes=num_class)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        input_size = x.shape[-2:]
        feats = self.encoder(x)
        spatial = self.fpn(feats)
        context = self.context(feats["c5"])
        boundary_feat, boundary_logits = self.boundary(spatial)
        fused = self.fusion(spatial, context, boundary_feat)

        main_logits = self.main_head(fused)
        main_logits = F.interpolate(main_logits, size=input_size, mode="bilinear", align_corners=False)

        if not return_aux:
            return main_logits

        boundary_logits = F.interpolate(boundary_logits, size=input_size, mode="bilinear", align_corners=False)
        aux_logits = self.aux_head(spatial)
        aux_logits = F.interpolate(aux_logits, size=input_size, mode="bilinear", align_corners=False)
        return {
            "out": main_logits,
            "boundary": boundary_logits,
            "aux": aux_logits,
        }

