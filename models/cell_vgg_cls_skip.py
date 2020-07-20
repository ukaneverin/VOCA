import torch.nn as nn
import math

cfg = {
    'config1': ['E', 64, 128, 256, 512, 512, 'D', 512, 512, 256, 128, 64],
    'config2': ['E', 64, 128, 256, 512, 'P', 'D', 512, 256, 128, 64],
    'config3': ['E', 64, 128, 256, 'D', 256, 128, 64],
    'config4': ['E', 64, 256, 'D', 256, 64],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class cell_vgg_cls_skip(nn.Module):

    def __init__(self, vgg_name, subnet_layer_number=2):
        super(cell_vgg_cls_skip, self).__init__()
        self.conv1 = self._mkconv(3,64)
        self.conv2 = self._mkconv(64,128)
        self.conv3 = self._mkconv(128,256)
        self.conv4 = self._mkconv(256,512)
        self.bottleneck = nn.Sequential(
                        nn.Conv2d(512, 512, kernel_size=3, padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True)
                    )
        self.deconv1 = self._mkdeconv(512,256)
        self.deconv2 = self._mkdeconv(256,128)
        self.deconv3 = self._mkdeconv(128,64)
        self.deconv4 = self._mkdeconv(64,64)

        self.classifier = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.regressor = nn.Conv2d(64, 2, kernel_size=1)
        self.quantifier = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU()
        )
        self._initialize_weights()

    def forward(self, x):

        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        d1 = c4 + self.bottleneck(c4)
        d2 = c3 + self.deconv1(d1)
        d3 = c2 + self.deconv2(d2)
        d4 = c1 + self.deconv3(d3)

        out = self.deconv4(d4)

        cls_map = self.classifier(out)

        reg_map = self.regressor(out)

        qtf_map = self.quantifier(out)

        return cls_map, reg_map, qtf_map

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def _mkconv(self, in_channels, out_channels):
        return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )

    def _mkdeconv(self, in_channels, out_channels):
        return nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
