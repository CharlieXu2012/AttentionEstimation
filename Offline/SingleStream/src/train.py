#!/usr/bin/env python3
import time
import copy
import torch
from torch.autograd import Variable

def train_network(net, dataloaders, dataset_sizes, batch_size, criterion, 
        optimizer, max_epochs, gpu):
    """Train network.

    Args:
        net (torchvision.models):   network to train
        dataloaders (dictionary):   contains torch.utils.data.DataLoader 
                                        for both training and validation
        dataset_sizes (dictionary): size of training and validation datasets
        batch_size (int):           size of mini-batch
        criterion (torch.nn.modules.loss):  loss function
        opitimier (torch.optim):    optimization algorithm.
        max_epochs (int):           max number of epochs used for training
        gpu (bool):                 gpu availability

    Returns:
        torchvision.models:     best trained model
        float:                  best validaion accuracy
        dictionary:             training and validation losses
        dictionary:             training and validation accuracy
    """
    # start timer
    start = time.time()
    # store network to GPU
    if gpu:
        net = net.cuda()

    # store best validation accuracy
    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    losses = {'Train': [], 'Valid': []}
    accuracies = {'Train': [], 'Valid': []}
    patience = 0
    for epoch in range(max_epochs):
        print()
        print('Epoch {}'.format(epoch))
        print('-' * 8)
        # each epoch has a training and validation phase
        for phase in ['Train', 'Valid']:
            if phase == 'Train':
                net.train(True)  # set model to training model
            else:
                net.train(False)  # set model to evaluation mode

            # used for accuracy and losses
            running_loss = 0
            running_correct = 0
            # iterate over data
            for i, data in enumerate(dataloaders[phase]):
                # get the inputs
                inputs, labels = data['X'], data['y']
                # reshape [numSeqs, batchSize, numChannels, Height, Width]
                inputs = inputs.transpose(0,1)
                # wrap in Variable
                if gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                inputs = Variable(inputs)
                labels = Variable(labels)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward pass
                outputs = net.forward(inputs)
                # loss + predicted
                _, pred = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                correct = torch.sum(pred == labels.data)
                # backwards + optimize only if in training phase
                if phase == 'Train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_correct += correct
                
            # find size of dataset (numBatches * batchSize)
            epoch_loss = running_loss * batch_size / dataset_sizes[phase]
            epoch_acc = running_correct / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            # store stats
            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc)
            if phase == 'Valid':
                patience += 1
                if epoch_acc > best_acc:
                    # deep copy the model
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(net.state_dict())
                    patience = 0

        if patience == 20:
            break

    # print elapsed time
    time_elapsed = time.time() - start
    print()
    print('Training Complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // (60*60), time_elapsed // 60 % 60
        ))
    # load the best model weights
    net.load_state_dict(best_model_wts)
    return net, best_acc, losses, accuracies
