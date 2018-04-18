import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

class SingleStream(nn.Module):
    """Single Stream (Spatial OR Temporal) + LSTM
    Args:
        model (string):     stream CNN architecture
        rnn_hidden (int):   number of hidden units in each rnn layer
        rnn_layers (int):   number of layers in rnn model
        pretrained (bool):  if model is pretrained with ImageNet
        finetuned (bool):   if model is finetuned with attention dataset
    """

    def __init__(self, model, rnn_hidden, rnn_layers, pretrained=True,
            finetuned=True):
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

        # add lstm layer
        self.lstm = nn.LSTM(num_fts, rnn_hidden, rnn_layers)
        # add linear layer
        self.fc = nn.Linear(rnn_hidden, 4)

    def forward(self, inputs):
        """Forward pass through network.
        Args:
            inputs (torch.Tensor): tensor of dimensions
                [numSeqs x batchSize x numChannels x Width x Height]
        Returns:
            torch.Tensor: final output of dimensions
                [batchSize x numClasses]
        """
        # list to hold features
        feats = []
        # for each input in sequence
        for inp in inputs:
            # pass through cnn
            outs = self.cnn.forward(inp).data
            outs = torch.squeeze(outs)
            feats.append(outs)
        
        # format features and store in Variable
        feats = torch.stack(feats)
        feats = Variable(feats)
        # pass through LSTM
        outputs, _ = self.lstm(feats)
        outputs = self.fc(outputs[-1])
        return outputs

def main():
    """Main Function."""
    model = 'VGGNet11'
    rnn_hidden = 32
    rnn_layers = 1
    net = SingleStream(model, rnn_hidden, rnn_layers, pretrained=False)
    print(net)

    # test model
    gpu = torch.cuda.is_available()
    sequence_len = 10
    window_size = 5
    batch_size = 2
    inputs = torch.randn(sequence_len, batch_size, 3, 244, 224)
    print('inputs:', inputs.shape)
    if gpu:
        net = net.cuda()
        inputs = Variable(inputs.cuda())

    else:
        inputs = Variable(inputs)

    # slide window though sequence
    for i in range(sequence_len-window_size):
        inp = inputs[i:i+window_size]
        print('inp:', inp.shape)
        # pass through network
        output = net.forward(inp)
        print('output:', output.size())

if __name__ == '__main__':
    main()
