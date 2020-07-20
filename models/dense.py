'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes, encoding_stage):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bnPool = nn.BatchNorm2d(out_planes)
        if encoding_stage:
            self.out = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=2, bias=False)
        else:
            self.out = nn.ConvTranspose2d(out_planes, out_planes, kernel_size=3, stride=2, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.out(F.relu(self.bnPool(out)))
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes, encoding_stage=True)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes, encoding_stage=True)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes, encoding_stage=True)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans4 = Transition(num_planes, out_planes, encoding_stage=True)
        num_planes = out_planes

        self.dense5 = self._make_dense_layers(block, num_planes, nblocks[4])
        num_planes += nblocks[4]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans5 = Transition(num_planes, out_planes, encoding_stage=False)
        num_planes = out_planes

        self.dense6 = self._make_dense_layers(block, num_planes, nblocks[5])
        num_planes += nblocks[5]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans6 = Transition(num_planes, out_planes, encoding_stage=False)
        num_planes = out_planes

        self.dense7 = self._make_dense_layers(block, num_planes, nblocks[6])
        num_planes += nblocks[6]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans7 = Transition(num_planes, out_planes, encoding_stage=False)
        num_planes = out_planes

        self.dense8 = self._make_dense_layers(block, num_planes, nblocks[7])
        num_planes += nblocks[7]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans8 = Transition(num_planes, out_planes, encoding_stage=False)
        num_planes = out_planes

        self.classifier = nn.Sequential(
            nn.Conv2d(num_planes, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.regressor = nn.Conv2d(num_planes, 2, kernel_size=1)
        self.quantifier = nn.Sequential(
            nn.Conv2d(num_planes, 1, kernel_size=1),
            nn.ReLU()
        )

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.trans4(self.dense4(out))
        out = self.trans5(self.dense5(out))
        out = self.trans6(self.dense6(out))
        out = self.trans7(self.dense7(out))
        out = self.trans8(self.dense8(out))

        cls_map = self.classifier(out)
        reg_map = self.regressor(out)
        qtf_map = self.quantifier(out)

        return cls_map, reg_map, qtf_map

def DenseNet121():
    return DenseNet(Bottleneck, [6,12,24,16,16,24,12,6], growth_rate=6)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

def densenet_cifar():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)

def test_densenet():
    net = densenet_cifar()
    x = torch.randn(1,3,32,32)
    y = net(Variable(x))
    print(y)

# test_densenet()
