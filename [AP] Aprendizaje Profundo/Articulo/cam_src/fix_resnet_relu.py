from torch import Tensor
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck

def basicblock_forward(self, x: Tensor) -> Tensor:
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = F.relu(out)

    return out

def bottleneck_forward(self, x: Tensor) -> Tensor:
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = F.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = F.relu(out)

    return out

BasicBlock.forward = basicblock_forward
Bottleneck.forward = bottleneck_forward