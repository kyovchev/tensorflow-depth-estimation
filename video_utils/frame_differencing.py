import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

frames=[]
MAX_FRAMES = 1000
N = 5
THRESH = 10
ASSIGN_VALUE = 255 #Value to assign the pixel if the threshold is met

cap = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.input))  #Capture using Computer's Webcam


sum = np.zeros((240, 320), dtype=np.float64)
count = 0

multiframes = []

for t in range(MAX_FRAMES):
    #Capture frame by frame
    ret, frame = cap.read()
    #Convert frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    sum += frame_gray.astype(np.float64)
    count = count + 1
    #Append to list of frames
    frames.append(frame_gray)
    mean = (sum / count).astype(np.uint8)
    detect = frame_gray.astype(np.float64) - mean.astype(np.float64)
    detect[detect < 10] = 0
    detect = ((detect / np.max(detect)) * 255).astype(np.uint8)
    multiframes.append(detect)

    if t >= N:
        #D(N) = || I(t) - I(t+N) || = || I(t-N) - I(t) ||
        diff = cv2.absdiff(frames[t-1], frames[t])
        #Mask Thresholding
        threshold_method = cv2.THRESH_BINARY
        ret, motion_mask = cv2.threshold(diff, THRESH, ASSIGN_VALUE, threshold_method)
        #Display the Motion Mask

        newdetect = np.zeros((240, 320), dtype=np.float64)
        for k in range(5):
            newdetect += multiframes[len(multiframes) - k - 1]
        newdetect = (newdetect / 5).astype(np.uint8)
        newdetect[newdetect < 64] = 0
        cv2.imshow('Frame', frame)
        cv2.imshow('Mean', (sum / count).astype(np.uint8))
        cv2.imshow('detect', detect)
        cv2.imshow('newdetect', newdetect)
        cv2.imshow('Motion Mask', motion_mask)

        keyboard = cv2.waitKey(100)
        if keyboard == 'q' or keyboard == 27:
            break