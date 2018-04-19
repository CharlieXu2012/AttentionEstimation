#!/usr/bin/env python3
import numbers
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms

class FlowDataset(Dataset):
    """Flow Features for Attention Level Dataset.
    Args:
        labels_path (string):   path to text file with annotations
        sequence_len (int):     length of video clip
        transform (callable):   transform to be applied on video sequence

    Returns:
        torch.utils.data.Dataset:   dataset object
    """ 
    def __init__(self, labels_path, sequence_len, transform=None):
        # read video paths and labels
        with open(labels_path, 'r') as f:
            data = f.read()
            data = data.split()
            data = np.array(data)
            data = np.reshape(data, (-1, 2))
        
        self.data = data
        self.sequence_len = sequence_len
        self.transform = transform
        self.cap = cv2.VideoCapture()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # create list to hold video data
        X = []
        y = int(self.data[idx, 1]) - 1
        video_path = 'data/' + self.data[idx, 0]
        # open video
        self.cap.open(video_path)
        # set to beginning of sequence
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 100-self.sequence_len)
        # read initial frame
        _, frame1 = self.cap.read()
        frame1 = cv2.resize(frame1, (256,256))
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        for i in range(self.sequence_len-1):
            _, frame2 = self.cap.read()
            frame2 = cv2.resize(frame2, (256,256))
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            # compute flow
            flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None,
                    0.5, 3, 15, 3, 5, 1.2, 0)
            # compute flow mag
            mag, _ = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])
            # normalize
            flow[:,:,0] = cv2.normalize(flow[:,:,0], None, 0, 255,
                    cv2.NORM_MINMAX)
            flow[:,:,1] = cv2.normalize(flow[:,:,1], None, 0, 255,
                    cv2.NORM_MINMAX)
            mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            # normalize between 0-1
            X.append(np.stack((flow[:,:,0], flow[:,:,1], mag),axis=2))
            frame1 = frame2

        # convert to numpy array
        X = np.array(X)
        # transform data
        if self.transform:
            X = self.transform(X)
        # reformat [numSeqs x numChannels x Height x Width]
        X = np.transpose(X, (0,3,1,2))
        # store in sample
        sample = {'X': X, 'y': y}
        return sample

class AppearanceDataset(Dataset):
    """Appearance Features for Attention Level Dataset.

    Args:
        labels_path (string):   path to text file with annotations
        sequence_len (int):     length of video clip
        transform (callable):   transform to be applied on video sequence
        
    Returns:
        torch.utils.data.Dataset:   dataset object
    """
    def __init__(self, labels_path, sequence_len, transform=None):
        # read video paths and labels
        with open(labels_path, 'r') as f:
            data = f.read()
            data = data.split()
            data = np.array(data)
            data = np.reshape(data, (-1, 2))
        
        self.data = data
        self.sequence_len = sequence_len
        self.transform = transform
        self.cap = cv2.VideoCapture()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # create list to hold video data
        X = []
        y = int(self.data[idx, 1]) - 1
        video_path = 'data/' + self.data[idx, 0]
        # open video
        self.cap.open(video_path)
        # set to beginning of sequence
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 100-self.sequence_len)
        for i in range(self.sequence_len):
            _, frame = self.cap.read()
            frame = cv2.resize(frame, (256,256))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            X.append(frame)

        # convert to numpy array
        X = np.array(X)
        # transform data
        if self.transform:
            X = self.transform(X)
        # reformat [numSeqs x numChannels x Height x Width]
        X = np.transpose(X, (0,3,1,2))
        # store in sample
        sample = {'X': X, 'y': y}
        return sample

class CenterCrop():
    """Crop frames in video sequence at the center.

    Args:
        output_size (tuple): Desired output size of crop.
    """
    
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, video):
        """
        Args:
            video (ndarray): Video to be center-cropped.
        
        Returns:
            ndarray: Center-cropped video.
        """
        # video dimensions
        h, w = video.shape[1:3]
        new_h, new_w = self.output_size
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        # center-crop each frame
        video_new = video[:, top:top+new_h, left:left+new_w, :]
        return video_new

class RandomCrop():
    """Crop randomly the frames in a video sequence.

    Args:
        output_size (tuple): Desired output size.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, video):
        """
        Args:
            video (ndarray): Video to be cropped.

        Returns:
            ndarray: Cropped video.
        """
        # video dimensions
        h, w = video.shape[1:3]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        # randomly crop each frame
        video_new = video[:, top:top+new_h, left:left+new_w, :]
        return video_new

class RandomHorizontalFlip():
    """Horizontally flip a video sequence.

    Args:
        p (float): Probability of image being flipped. Default value is 0.5.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video):
        """
        Args:
            video (ndarray): Video to be flipped.

        Returns:
            ndarray: Randomly flipped video.
        """
        # check to perform flip
        if np.random.random_sample() < self.p:
            # flip video
            video_new = np.flip(video, 2)
            return video_new

        return video

class RandomRotation():
    """Rotate video sequence by an angle.

    Args:
        degrees (float or int): Range of degrees to select from.
    """
    
    def __init__(self, degrees):
        assert isinstance(degrees, numbers.Real)
        self.degrees = degrees

    def __call__(self, video):
        """
        Args:
            video (ndarray): Video to be rotated.

        Returns:
            ndarray: Randomly rotated video.
        """
        # hold transformed video
        video_new = np.zeros_like(video)
        h, w = video.shape[1:3]
        # random rotation
        angle = np.random.uniform(-self.degrees, self.degrees)
        # create rotation matrix with center point at the center of frame
        M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1)
        # rotate each frame
        for idx, frame in enumerate(video):
            video_new[idx] = cv2.warpAffine(frame, M, (w,h))
        
        return video_new

class RandomWindow():
    """Window video sequence.

    Args:
        window_size (int):   Length of window..
    """
    def __init__(self, window_size):
        assert isinstance(window_size, int)
        self.window_size = window_size

    def __call__(self, video):
        """
        Args:
            video (ndarray):    Video to be sampled.

        Returns:
            ndarray: Randomly sampled video.
        """
        print('Video:', video.shape)
        if self.window_size >= video.shape[0]:
            return video

        start = np.random.randint(video.shape[0] - self.window_size)
        video_new = video[start:start+self.window_size]
        return video_new

class Normalize():
    """Normalize video with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this
    transform will normalize each channge of the input video.

    Args:
        mean (list): Sequence of means for each channel.
        std (list): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        assert isinstance(mean, list)
        assert isinstance(std, list)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, video):
        """
        Args:
            video (ndarray): Video to be normalized

        Returns:
            ndarray: Normalized video.
        """
        video = video / 255
        video = (video - self.mean) / self.std
        video = np.asarray(video, dtype=np.float32)
        return video

def count_labels(data_path):
    """Count the number of instances in each class.

    Args:
        data_path (string):     Path to annotations.

    Returns:
        counts (ndarray):       array containing number of instances.
    """
    counts = np.zeros(4, dtype=int)
    with open(data_path, 'r') as f:
        for line in f:
            line = int(line.split()[1]) - 1
            counts[line] += 1

    return counts

def get_loaders(train_path, valid_path, batch_size, sequence_len, flow,
        num_workers, gpu):
    """Return dictionary of torch.utils.data.DataLoader.

    Args:
        train_path (string):    path to training annotations
        valid_path (string):    path to validation annotations
        batch_size (int):       number of instances in batch
        sequence_len (int):     number of frames to keep in video (counting
                                    from last frame)
        flow (bool):            use flow or appearance dataset
        num_workers (int):      number of subprocesses used for data loading
        gpu (bool):             presence of gpu

    Returns:
        torch.utils.data.DataLoader:    dataloader for custom dataset
        dictionary:                     dataset length for training and 
                                            validation
    """
    # data augmentation and normalization for training
    # just normalization for validation
    data_transforms = {
            'Train': transforms.Compose([
                RandomCrop((224,224)),
                RandomHorizontalFlip(),
                RandomRotation(15),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            'Valid': transforms.Compose([
                CenterCrop((224,224)),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            }

    # create dataset object
    if flow:
        datasets = {
                'Train': FlowDataset(train_path, sequence_len,
                    data_transforms['Train']),
                'Valid': FlowDataset(valid_path, sequence_len,
                    data_transforms['Valid'])
                }
    else:
        datasets = {
                'Train': AppearanceDataset(train_path, sequence_len,
                    data_transforms['Train']),
                'Valid': AppearanceDataset(valid_path, sequence_len,
                    data_transforms['Valid'])
                }

    # dataset sizes
    dataset_sizes = {'Train': len(datasets['Train']),
                     'Valid': len(datasets['Valid'])}

    # add weighted sampler since unbalanced dataset
    c = count_labels(train_path)
    weights = np.zeros(dataset_sizes['Train'])
    weights[: c[0]] = c[1] / c[0]
    weights[c[0]: c[0]+c[1]] = 1
    weights[c[0]+c[1]:c[0]+c[1]+c[2]] = c[1] / c[2]
    weights[c[0]+c[1]+c[2]:] = c[1] / c[3]

    # create dataloders
    dataloaders = {
            'Train': DataLoader(datasets['Train'], batch_size=batch_size,
                sampler=WeightedRandomSampler(weights, dataset_sizes['Train'])),
            'Valid': DataLoader(datasets['Valid'], batch_size=batch_size,
                shuffle=True)
            }

    return dataloaders, dataset_sizes

def main():
    """Main Function."""
    import matplotlib.pyplot as plt
    from torchvision import utils

    # data paths
    train_path = 'data/train_data.txt'
    valid_path = 'data/valid_data.txt'
    test_path = 'data/test_data.txt'
    # hyper-parameters
    batch_size = 2
    sequence_len = 24  # counting backwards from last frame
    
    dataloaders, dataset_sizes = get_loaders(train_path, valid_path, 
            batch_size, sequence_len, flow=True, num_workers=0, gpu=False)
    print('Dataset Sizes:')
    print(dataset_sizes)

    def imshow(grid):
        grid = grid.numpy().transpose((1,2,0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        grid = std * grid + mean
        grid = np.clip(grid, 0, 1)
        plt.imshow(grid, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.show()

    train_batch = next(iter(dataloaders['Valid']))
    data, labels = train_batch['X'], train_batch['y']
    print('data:', data.shape)
    print('labels:', labels.shape)
    grid = utils.make_grid(data[0])
    imshow(grid)

if __name__ == '__main__':
    main()
