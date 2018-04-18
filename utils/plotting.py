import matplotlib.pyplot as plt

def plot_data(losses, accuracies, name):
    """Plot training and validation statistics.
    Args:
        losses (dictionary): containing list of cross entrophy losses for
                                training and validation splits
        accuracies (dictionary): contains list of accuracies for training
                                    and validation splits
        name (string): name to save plot
    """
    # convert accuracies to percentages
    accuracies['Train'] = [acc * 100 for acc in accuracies['Train']]
    accuracies['Valid'] = [acc * 100 for acc in accuracies['Valid']]
    # set fontsize
    plt.rcParams.update({'font.size': 13})
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8,5))
    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('Cross Entropy Loss')
    ax1.set_ylim(0,2)
    ax1.plot(losses['Train'], label='Training')
    ax1.plot(losses['Valid'], label='Validation')
    ax1.legend(loc='upper right')

    ax2.set_xlabel('Number of Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0,100)
    ax2.plot(accuracies['Train'], label='Training')
    ax2.plot(accuracies['Valid'], label='Validation')
    ax2.legend(loc='upper left')

    fig.tight_layout()
    fig.savefig(name)
