#__author__ = Yossi Adi
import torch.nn as nn
import torch.nn.functional as F

def _make_layers(cfg, kernel=3):
    layers = []
    in_channels = 1
    for x in cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, x, kernel_size=kernel, padding=1),
                       nn.BatchNorm2d(x),
                       nn.ReLU(inplace=True)]
            in_channels = x
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.Sequential(*layers)

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, class_num=30):
        super(VGG, self).__init__()
        self.features = _make_layers(cfg[vgg_name])
        self.fc1 = nn.Linear(7680, 512)
        self.fc2 = nn.Linear(512, class_num)

    def forward(self, x):
        # out = self.features(x)
        for m in self.features.children():
            x = m(x)

        out = x
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)




