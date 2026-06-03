"""
ResNet (Residual Network) — Complete Implementation
====================================================

Implements ResNet-18, 34, 50, and 101 from scratch, including both BasicBlock
(for shallower variants) and Bottleneck (for deeper variants).

Reference: "Deep Residual Learning for Image Recognition" (He et al., 2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Two 3x3 conv layers with a residual (skip) connection.

    Used in ResNet-18 and ResNet-34. The expansion factor is 1, meaning the
    output channels equal the internal channels.
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return F.relu(out)


class Bottleneck(nn.Module):
    """1x1 -> 3x3 -> 1x1 conv layers with a residual connection.

    Used in ResNet-50, 101, and 152. The 1x1 convs reduce and restore the
    channel dimension, creating a "bottleneck" that makes the 3x3 conv cheaper.
    Expansion is 4: output channels = 4 * out_channels.
    """

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return F.relu(out)


class ResNet(nn.Module):
    """Generic ResNet that supports any block type and layer configuration.

    Architecture:
        conv1 (7x7, stride 2) -> bn -> relu -> maxpool (3x3, stride 2)
        -> layer1 -> layer2 -> layer3 -> layer4
        -> adaptive avgpool -> fc

    The spatial resolution halves at conv1 (stride 2), maxpool (stride 2),
    and at the first block of layers 2-4 (stride 2). For 224x224 input:
    224 -> 112 -> 56 -> 28 -> 14 -> 7 -> 1 (after avgpool).
    """

    def __init__(self, block, layers, num_classes=1000):
        """
        Args:
            block: BasicBlock or Bottleneck
            layers: list of 4 ints, number of blocks per stage
            num_classes: number of output classes
        """
        super().__init__()
        self.in_channels = 64

        # Initial convolution: large 7x7 kernel captures low-level features
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Four residual stages with increasing channels: 64, 128, 256, 512
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """Build one residual stage.

        The first block may downsample spatially (stride > 1) and change
        channel dimensions. All subsequent blocks preserve dimensions.
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Kaiming initialization, standard for networks with ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))   # (B, 64, 112, 112)
        x = self.maxpool(x)                    # (B, 64, 56, 56)

        x = self.layer1(x)                     # (B, 64*exp, 56, 56)
        x = self.layer2(x)                     # (B, 128*exp, 28, 28)
        x = self.layer3(x)                     # (B, 256*exp, 14, 14)
        x = self.layer4(x)                     # (B, 512*exp, 7, 7)

        x = self.avgpool(x)                    # (B, 512*exp, 1, 1)
        x = torch.flatten(x, 1)               # (B, 512*exp)
        x = self.fc(x)                         # (B, num_classes)
        return x


# ---------------------------------------------------------------------------
# Factory functions for standard ResNet configurations
# ---------------------------------------------------------------------------

def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


# ---------------------------------------------------------------------------
# Pre-Activation ResNet variant (BN -> ReLU -> Conv)
# ---------------------------------------------------------------------------

class PreActBasicBlock(nn.Module):
    """Pre-activation BasicBlock: BN-ReLU-Conv instead of Conv-BN-ReLU.

    The skip connection is a true identity mapping (no normalization on the
    shortcut), which improves gradient flow in very deep networks.
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(x))
        if self.downsample is not None:
            identity = self.downsample(out)
        out = self.conv1(out)

        out = self.conv2(F.relu(self.bn2(out)))
        out += identity
        return out


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    x = torch.randn(2, 3, 224, 224)

    configs = {
        "ResNet-18": resnet18(num_classes=10),
        "ResNet-34": resnet34(num_classes=10),
        "ResNet-50": resnet50(num_classes=10),
        "ResNet-101": resnet101(num_classes=10),
    }

    for name, model in configs.items():
        model.eval()
        with torch.no_grad():
            out = model(x)
        print(f"{name:12s} | params: {count_params(model):>12,} | "
              f"input: {list(x.shape)} -> output: {list(out.shape)}")

    # Quick sanity check on pre-activation block
    block = PreActBasicBlock(64, 64)
    block.eval()
    with torch.no_grad():
        t = torch.randn(1, 64, 56, 56)
        print(f"\nPreActBasicBlock: {list(t.shape)} -> {list(block(t).shape)}")

    print("\nAll ResNet variants verified successfully!")
