#!/usr/bin/env python3
import torch.nn as nn
import torchvision.models as models

class SingleFrame(nn.Module):
    """Single Frame Model
    Args:
        model (string): stream CNN architecture
        pretrained (bool):  if model is pretrained with ImageNet (default true)
        finetuned (bool):   if model is finetuned (default true)
    """

    def __init__(self, model, pretrained=True, finetuned=True):
        super().__init__()
        if model is 'AlexNet':
            self.cnn = models.alexnet(pretrained)
            num_fts = self.cnn.classifier[4].in_features
            self.cnn.classifier = nn.Sequential(
                    *list(self.cnn.classifier.children())[:-3]
                    )
        elif model is 'VGGNet11':
            self.cnn = models.vgg11_bn(pretrained)
            num_fts = self.cnn.classifier[3].in_features
            self.cnn.classifier = nn.Sequential(
                    *list(self.cnn.classifier.children())[:-4]
                    )
        elif model is 'VGGNet16':
            self.cnn = models.vgg16_bn(pretrained)
            num_fts = self.cnn.classifier[3].in_features
            self.cnn.classifier = nn.Sequential(
                    *list(self.cnn.classifier.children())[:-4]
                    )
        elif model is 'VGGNet19':
            self.cnn = models.vgg19_bn(pretrained)
            num_fts = self.cnn.classifier[3].in_features
            self.cnn.classifier = nn.Sequential(
                    *list(self.cnn.classifier.children())[:-4]
                    )
        elif model is 'ResNet18':
            self.cnn = models.resnet18(pretrained)
            num_fts = self.cnn.fc.in_features
            self.cnn = nn.Sequential(
                    *list(self.cnn.children())[:-1]
                    )
        elif model is 'ResNet34':
            self.cnn = models.resnet34(pretrained)
            num_fts = self.cnn.fc.in_features
            self.cnn = nn.Sequential(
                    *list(self.cnn.children())[:-1]
                    )
        else:
            print('Please input correct model architecture')
            return

        for param in self.cnn.parameters():
            param.requires_grad = finetuned

        # add fc layer
        self.fc = nn.Linear(num_fts, 4)
        
    def forward(self, inp):
        out = self.cnn.forward(inp)
        out = self.fc(out)
        return out

def main():
    """Test Function."""
    model = 'VGGNet11'
    net = SingleFrame(model, pretrained=False)
    print(net)

if __name__ == '__main__':
    main()
