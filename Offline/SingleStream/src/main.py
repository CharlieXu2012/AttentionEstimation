#!/usr/bin/env python3
import torch
import torch.nn as nn
from dataloader import get_loaders
from model import SingleStream
from train import train_network
import sys
sys.path.insert(0, 'utils')
from plotting import plot_data

def main():
    """Main Function."""
    # dataloaders parameters
    gpu = torch.cuda.is_available()
    train_path = 'data/train_data.txt'
    valid_path = 'data/valid_data.txt'
    test_path = 'data/test_data.txt'
    batch_size = 32
    sequence_len = 100  # counting backwards from last frame
    sample_rate = 5
    flow = False
    num_workers = 2
    # network parameters
    model = 'VGGNet19'
    rnn_hidden = 512
    rnn_layers = 2
    # training parameters
    max_epochs = 1
    learning_rate = 1e-4
    criterion = nn.CrossEntropyLoss()

    # create dataloaders
    dataloaders, dataset_sizes = get_loaders(train_path, valid_path,
            batch_size, sequence_len, sample_rate, flow, num_workers,
            gpu=True)
    print('Dataset Sizes:')
    print(dataset_sizes)
    # create network object and optimizer
    net = SingleStream(model, rnn_hidden, rnn_layers, pretrained=True)
    print(net)
    optimizer = torch.optim.Adam(net.parameters(), learning_rate)
    # train the network
    net, val_acc, losses, accuracies = train_network(net, dataloaders, 
            dataset_sizes, batch_size, criterion, optimizer, max_epochs, gpu)
    # plot
    plot_data(losses, accuracies, 'outputs/offline/SingleStreamPlots.png')
    # save network
    torch.save(net.state_dict(), 'outputs/offline/SingleStreamParams.pkl')

if __name__ == '__main__':
    main()

