'''DenseNet in PyTorch.'''
import math

import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseNet(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNet, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        self.features = nn.Sequential(*list(self.densenet.children())[:-1])
        # Replace the classifier with a new one for the desired number of classes
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        #out = self.features(x) 
        #out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        #out = out.view(out.size(0), -1)
        #out = self.densenet.classifier(out)
        return self.densenet(x)


def test():
    net = DenseNet(num_classes=101)
    x = torch.randn(1,3,224,224)
    y = net(x)
    #print(net)
    print(y)

test()