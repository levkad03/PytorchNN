import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

pretrained_model = models.alexnet(pretrained=True)

for param in pretrained_model.parameters():
    param.requires_grad = False


class ConvNet(nn.Module):
    def __init__(self, num_classes=15):
        super(ConvNet, self).__init__()

        self.pretrained_features = pretrained_model.features
        self.avgpool = pretrained_model.avgpool

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        out = self.pretrained_features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out
