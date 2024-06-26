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
            nn.Linear(4096, num_classes)
        )

        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2)
        # )
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(64, 192, kernel_size=5, padding=2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2)
        # )
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(192, 384, kernel_size=3, padding=1),
        #     nn.ReLU(),
        # )
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(384, 256, kernel_size=3, padding=1),
        #     nn.ReLU(),
        # )
        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2)
        # )
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # self.fc = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(256*6*6, 4096),
        #     nn.ReLU()
        # )
        # self.fc1 = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU()
        # )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(4096, num_classes),
        # )

    def forward(self, x):

        out = self.pretrained_features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        # out = self.layer1(x)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        # out = self.layer5(out)
        # out = self.avgpool(out)
        # out = torch.flatten(out, 1)
        # out = self.fc(out)
        # out = self.fc1(out)
        # out = self.fc2(out)
        return out

        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.dropout(x)
        # x = self.pool(F.relu(self.conv3(x)))
        # x = x.view(-1, 22 * 22 * 30)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)

        # return x
