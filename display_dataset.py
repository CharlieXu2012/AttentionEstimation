import cv2
import numpy as np

labels_path = 'data/valid_data.txt'
fps = 20
cap = cv2.VideoCapture()

with open(labels_path) as f:
    try:
        data = f.read().split()
        data = np.reshape(data, (-1,2))
        for instance in data:
            video_path, label = instance
            cap.open('data/' + video_path)
            for i in range(100):
                _, frame = cap.read()
                s = label + '-' + str(i+1)
                cv2.putText(frame, s, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0,0,225), 10)
                cv2.imshow('Frame', frame)
                cv2.waitKey(int(1/fps*1000))
            

    except KeyboardInterrupt:
        print('Program Closed.')
        f.close()
