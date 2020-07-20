'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pdb import set_trace

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, encoding_stage=True):
        super(BasicBlock, self).__init__()
        if stride != 1:
            if encoding_stage:
                self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, bias=False)
            else:
                self.conv1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=3, stride=stride, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if encoding_stage:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=3, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.ConvTranspose2d(in_planes, self.expansion*planes, kernel_size=3, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, encoding_stage=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if stride != 1:
            if encoding_stage:
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, bias=False)
            else:
                self.conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=stride, bias=False)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if encoding_stage:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=3, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.ConvTranspose2d(in_planes, self.expansion*planes, kernel_size=3, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2, encoding_stage=True)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, encoding_stage=True)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, encoding_stage=True)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, encoding_stage=True)

        self.layer5 = self._make_layer(block, 512, num_blocks[4], stride=2, encoding_stage=False)
        self.layer6 = self._make_layer(block, 256, num_blocks[5], stride=2, encoding_stage=False)
        self.layer7 = self._make_layer(block, 128, num_blocks[6], stride=2, encoding_stage=False)
        self.layer8 = self._make_layer(block, 64, num_blocks[7], stride=2, encoding_stage=False)

        self.classifier = nn.Sequential(
            nn.Conv2d(64*block.expansion, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
        self.regressor = nn.Conv2d(64*block.expansion, 2*num_classes, kernel_size=1)
        self.quantifier = nn.Sequential(
            nn.Conv2d(64*block.expansion, num_classes, kernel_size=1),
            nn.ReLU()
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.thresholder = nn.Linear(512+64, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride, encoding_stage):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, encoding_stage))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        h1 = self.avg_pool(out)

        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        h2 = self.avg_pool(out)

        cls_map = self.classifier(out)
        reg_map = self.regressor(out)
        qtf_map = self.quantifier(out)

        t = self.thresholder(torch.cat((h1,h2),1).view(h1.size(0),-1))

        return cls_map, reg_map, qtf_map, t, out


def ResNet18():
    return ResNet(BasicBlock, [3,4,6,3,3,6,4,3])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()
