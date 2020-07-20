import torch.nn as nn
import math

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M'], #, 512, 512, 'M', 512, 512, 'M'],
    'sub': [512, 512, 'M', 512, 512, 'M']
}

class cell_vgg_subnets_early(nn.Module):

    def __init__(self, vgg_name, num_classes=1000):
        super(cell_vgg_subnets_early, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.sub = self._make_layers(cfg['sub'])
        self.classifier = nn.Linear(512, num_classes)
        self.regressor = nn.Linear(512, 2)
        self._initialize_weights()

    def forward(self, x):

        x = self.features(x)

        #regression path
        reg_x = self.sub(x)
        reg_x = reg_x.view(reg_x.size(0), -1)
        trans_regs = self.regressor(reg_x)

        #classification path
        cls_x =self.sub(x)
        cls_x = cls_x.view(cls_x.size(0), -1)
        class_preds = self.classifier(cls_x)
        return class_preds, trans_regs

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

    def _make_layers(self, cfg):
        layers = []
        if cfg[0] == 64:
            in_channels = 3
        else:
            in_channels = 256
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
