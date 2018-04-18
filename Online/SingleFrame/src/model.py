#!/usr/bin/env python3
import torch.nn as nn
import torchvision.models as models

class SingleFrame(nn.Module):
    """Single Frame Classifier."""
    def __init__(self):
        super().__init__()
        self.cnn = models.vgg19_bn(pretrained=True)
        # number of features in fc1
        num_ftrs = self.cnn.classifier[3].in_features
        # remove last two fc layers (+ ReLU and Dropout layers)
        self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-4]
                )
        self.fc = nn.Linear(num_ftrs, 4)

    def forward(self, inp):
        out = self.cnn.forward(inp)
        out = self.fc(out)
        return out

def main():
    """Test Function."""
    net = SingleFrame()
    print(net)

if __name__ == '__main__':
    main()
