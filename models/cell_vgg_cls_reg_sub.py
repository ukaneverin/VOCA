import torch.nn as nn
import math

cfg = {
    'config1': ['E', 64, 128, 256, 512, 512, 'D', 512, 512, 256, 128, 64],
    'config2': ['E', 64, 128, 256, 512, 'D', 512, 256, 128, 64],
    'config3': ['E', 64, 128, 256, 'D', 256, 128, 64],
    'config4': ['E', 64, 256, 'D', 256, 64],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class cell_vgg_cls_reg_sub(nn.Module):

    def __init__(self, vgg_name, subnet_layer_number=2):
        super(cell_vgg_cls_reg_sub, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.subnet = self._make_subnets(2)
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
        x = self.features(x)

        cls_map = self.subnet(x)
        cls_map = self.classifier(cls_map)

        reg_map = self.subnet(x)
        reg_map = self.regressor(reg_map)

        qtf_map = self.subnet(x)
        qtf_map = self.quantifier(qtf_map)

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


    def _make_subnets(self, subnet_layer_number):
        layers = []
        for i in range(subnet_layer_number):
            layers += [nn.Conv2d(64, 64, kernel_size=3, padding=1),
                       nn.BatchNorm2d(64),
                       nn.ReLU(inplace=True)]

        return nn.Sequential(*layers)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        phase = 'encoding'
        for x in cfg:
            if x == 'E':
                phase = 'encoding'
            elif x == 'D':
                phase = 'decoding'
            elif x == 'P':
                phase = 'feature_pooling'
            else:
                if phase == 'encoding':
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, stride=2),
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                               # nn.Conv2d(x, x, kernel_size=2, stride=2),
                               # nn.BatchNorm2d(x),
                               # nn.ReLU(inplace=True)]
                    in_channels = x
                elif phase == 'decoding':
                    layers += [nn.ConvTranspose2d(in_channels, x, kernel_size=3, stride=2),
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                               # nn.Conv2d(x, x, kernel_size=3, padding=1),
                               # nn.BatchNorm2d(x),
                               # nn.ReLU(inplace=True)]
                    in_channels = x
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=1),
                               nn.BatchNorm2d(x),
                               nn.Sigmoid()]
                    in_channels = x

        return nn.Sequential(*layers)
