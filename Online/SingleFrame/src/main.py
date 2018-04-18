#!/usr/bin/env python3
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataloader import get_loaders
from train import train_network
from model import SingleFrame
from utils import plot_data

def main():
    """Main Function."""
    # dataloader parameters
    gpu = torch.cuda.is_available()
    train_path = 'data/train_data.txt'
    valid_path = 'data/valid_data.txt'
    batch_size = 32
    sequence_len = 50
    num_workers = 2
    # training parameters
    max_epochs = 200
    learning_rate = 1e-4
    criterion = nn.CrossEntropyLoss()

    # get dataloaders    
    dataloaders, dataset_sizes = get_loaders(train_path, valid_path,
            batch_size, sequence_len, num_workers, gpu)

    # create network and optimizier
    net = SingleFrame()
    print(net)
    optimizer = torch.optim.Adam(net.parameters(), learning_rate)
    # train the network
    net, val_acc = train_network(net, dataloaders, dataset_sizes, batch_size, 
            sequence_len, criterion, optimizer, max_epochs, gpu)
    print('Best Validation Acc:', val_acc)
    # plot
    plot_data(losses, accuracies, 'outputs/SingleFramePlots.png')
    # save network
    torch.save(net.state_dict(), 'outputs/SingleFrameParams.pkl')

if __name__ == '__main__':
    main()
