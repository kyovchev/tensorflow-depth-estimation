import glob
import numpy as np
import tensorflow as tf
import cv2
import imutils
import os

FRAMES_FOLDER = "video_4_cut_1"
OUTPUT_FOLDER = "video_4_cut_1_gray_depth"

ROOT_FOLDER = "D:/tensorflow-depth-estimation"

USE_EMPTY_ROAD_IMAGE = False
EMPTY_ROAD_IMAGE = f"{ROOT_FOLDER}/video_utils/videos/video_1_cut/frame_00953.jpg"

WIDTH = 320
HEIGHT = 240

IMAGE_W = 1920
IMAGE_H = 1080
WINDOW_W = 928
WINDOW_H = 696

# for video_1:
# WINDOW_X = (IMAGE_W - WINDOW_W) // 2
# WINDOW_Y = 0
# HEATMAP_Y = 0

# for video_2, video_3 and video_4:
WINDOW_X = 358
WINDOW_Y = 234
HEATMAP_Y = 100

MAX_D = 101
ADDED_D = 4

# MODEL_NAME = "model_unet_1708465933.keras" # trained on NYUv2
MODEL_NAME = "model_kitti_unet_1709064377.keras"

def preprocess_image(image, h=128, w=128, depth=False, horizontal_flip=False):
    image = image.copy()
    if depth:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, height=h)
    image = image.astype("float")
    image_w = image.shape[1]
    image = image[:, (image_w - w) // 2 : (w + image_w) // 2]

    if horizontal_flip:
        image = cv2.flip(image, 1)

    if depth:
        image = np.reshape(image, (image.shape[0], image.shape[1], 1))

    return (image - image.min()) / (image.max() - image.min())

model = tf.keras.models.load_model(f"{ROOT_FOLDER}/{MODEL_NAME}", compile=False) # only for prediction

if not os.path.isdir(f"{ROOT_FOLDER}/video_utils/videos/{OUTPUT_FOLDER}"):
    os.makedirs(f"{ROOT_FOLDER}/video_utils/videos/{OUTPUT_FOLDER}", exist_ok=True)

if USE_EMPTY_ROAD_IMAGE:
    image = cv2.imread(EMPTY_ROAD_IMAGE)
    image2 = image[WINDOW_Y : WINDOW_Y + WINDOW_H, WINDOW_X : WINDOW_X + WINDOW_W, :]

    x_test = np.empty((1, HEIGHT, WIDTH, 3))
    x_test[0, ] = preprocess_image(image2, HEIGHT, WIDTH, depth=False, horizontal_flip=False)

    empty_road_prediction = model.predict(x_test, batch_size=1)
    empty_road_prediction = empty_road_prediction[0]
    empty_road_prediction = ADDED_D + empty_road_prediction * MAX_D

for filename in glob.glob(f"{ROOT_FOLDER}/video_utils/videos/{FRAMES_FOLDER}/*.jpg"):
    image = cv2.imread(filename)
    image2 = image[WINDOW_Y : WINDOW_Y + WINDOW_H, WINDOW_X : WINDOW_X + WINDOW_W, :]

    x_test = np.empty((1, HEIGHT, WIDTH, 3))
    x_test[0, ] = preprocess_image(image2, HEIGHT, WIDTH, depth=False, horizontal_flip=False)

    prediction = model.predict(x_test, batch_size=1)
    prediction = prediction[0]
    prediction = ADDED_D + prediction * MAX_D

    grayscale = (255 - (prediction - ADDED_D) / MAX_D * 255).astype(np.uint8)

    cv2.imwrite(f"{ROOT_FOLDER}/video_utils/videos/{OUTPUT_FOLDER}/{filename.rsplit(os.sep, 1)[1]}", grayscale)