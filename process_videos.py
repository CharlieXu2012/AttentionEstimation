import os
import numpy as np
import cv2

def offline_format(dataset, sample_rate):
    """Read videos in dataset and save to disk as ndarrays.
    
    Args:
        dataset (string):   path to dataset text file
        sample_rate (int):  sample video
    """
    cap = cv2.VideoCapture()
    with open(dataset, 'r') as f:
        data = f.read()
        data = data.split()
        data = np.array(data)
        data = np.reshape(data, (-1, 2))
    # for each video in dataset
    for instance in data:
        # list to store frame data
        X = []
        video_path, _ = instance
        video_path = 'data/' + video_path
        cap.open(video_path)
        # store frames
        for i in range(100):
            _, frame = cap.read()
            frame = cv2.resize(frame, (256,256))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            X.append(frame)

        X = np.array(X)
        X = X[::sample_rate]
        video_path = video_path[:5] + 'offline/' + video_path[5:-3] + 'npy'
        np.save(video_path, X)
        print(video_path)

def online_format(dataset, sequence_start):
    """Read videos in dataset and save processed videos for online training
    to disk as ndarrays.

    Args:
        dataset (string):       path to dataset text file
        sequence_start (int):   start sequence from this index
    """
    cap = cv2.VideoCapture()
    with open(dataset, 'r') as f:
        data = f.read()
        data = data.split()
        data = np.array(data)
        data = np.reshape(data, (-1, 2))
    # for each video in dataset
    for instance in data:
        # list to store frame data
        X = []
        video_path, _ = instance
        video_path = 'data/' + video_path
        cap.open(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, sequence_start)
        for i in range(100-sequence_start):
            _, frame = cap.read()
            frame = cv2.resize(frame, (256,256))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            X.append(frame)

        X = np.array(X)
        video_path = video_path[:5] + 'online/' + video_path[5:-3] + 'npy'
        np.save(video_path, X)
        print(video_path)

def main():
    """Main Function."""
    # create directories
    os.system('mkdir data/offline')
    os.system('mkdir data/offline/positive')
    os.system('mkdir data/offline/negative')
    os.system('mkdir data/online')
    os.system('mkdir data/online/positive')
    os.system('mkdir data/online/negative')
    # data paths
    train_data = 'data/train_data.txt'
    valid_data = 'data/valid_data.txt'
    test_data = 'data/test_data.txt'
    sample_rate = 5
    sequence_start = 50  # start online sequence at frame 50

    datasets = [train_data, valid_data, test_data]
    for dataset in datasets:
#        offline_format(dataset, sample_rate)
        online_format(dataset, sequence_start)

if __name__ == '__main__':
    main()

