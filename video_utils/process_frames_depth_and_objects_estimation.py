import glob
import numpy as np
import tensorflow as tf
import cv2
import imutils
import os

FRAMES_FOLDER = "video_1_cut"
OUTPUT_FOLDER = "video_1_cut_kitti_unet_depth_object"

ROOT_FOLDER = "D:/tensorflow-depth-estimation"

EMPTY_ROAD_IMAGE = f"{ROOT_FOLDER}/video_utils/videos/video_1_cut/frame_00953.jpg"

WIDTH = 320
HEIGHT = 240

IMAGE_W = 1920
IMAGE_H = 1080
WINDOW_W = 928
WINDOW_H = 696
WINDOW_X = (IMAGE_W - WINDOW_W) // 2
MAX_D = 101
ADDED_D = 4
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

image = cv2.imread(EMPTY_ROAD_IMAGE)
image2 = image[:WINDOW_H, (IMAGE_W - WINDOW_W) // 2 : (IMAGE_W + WINDOW_W) // 2, :]

x_test = np.empty((1, HEIGHT, WIDTH, 3))
x_test[0, ] = preprocess_image(image2, HEIGHT, WIDTH, depth=False, horizontal_flip=False)

empty_road_prediction = model.predict(x_test, batch_size=1)
empty_road_prediction = empty_road_prediction[0]
empty_road_prediction = ADDED_D + empty_road_prediction * MAX_D

for filename in glob.glob(f"{ROOT_FOLDER}/video_utils/videos/{FRAMES_FOLDER}/*.jpg"):
    image = cv2.imread(filename)
    image2 = image[:WINDOW_H, (IMAGE_W - WINDOW_W) // 2 : (IMAGE_W + WINDOW_W) // 2, :]

    x_test = np.empty((1, HEIGHT, WIDTH, 3))
    x_test[0, ] = preprocess_image(image2, HEIGHT, WIDTH, depth=False, horizontal_flip=False)

    prediction = model.predict(x_test, batch_size=1)
    prediction = prediction[0]
    prediction = ADDED_D + prediction * MAX_D

    depth3 = image.copy()

    heatmap = cv2.applyColorMap((255 - (prediction - ADDED_D) / MAX_D * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heatmap2 = cv2.resize(heatmap, (WINDOW_W, WINDOW_H))
    alpha = 0.1
    depth3[:WINDOW_H, (IMAGE_W - WINDOW_W) // 2 : (IMAGE_W + WINDOW_W) // 2] = \
        cv2.addWeighted(depth3[:WINDOW_H, (IMAGE_W - WINDOW_W) // 2 : (IMAGE_W + WINDOW_W) // 2], alpha, heatmap2[:, :], 1 - alpha, 0)

    diff = empty_road_prediction[:,:,0] / prediction[:,:,0]
    diff = diff > 1.25
    depth = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for h in range(60, 230):
        for w in range(5, 315):
            if h < 220 and (w / (h - 220) > -0.7 or (WIDTH - w) / (h - 220) > -0.5):
                continue
            if diff[h, w]:
                depth[h, w] = 255
    ret1, lab = cv2.connectedComponents(depth)

    points = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    for i in range(1, ret1):
        component = np.zeros((HEIGHT, WIDTH), np.uint8)
        component[lab == i] = 255
        if np.count_nonzero(component) > 100:
            rect = cv2.boundingRect(component)
            points[rect[1] + rect[3], rect[0] + rect[2] // 2] = 255

    font = cv2.FONT_HERSHEY_SIMPLEX 
    font_scale = 0.7
    color = (255, 0, 0) 
    thickness = 1

    find = points.copy()

    for h in range(200, 60, -1):
        for w in range(WIDTH):
            if find[h, w] > 128:
                dist = prediction[h, w, 0]
                text = f"{dist:.2f} m"
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_pos = (int(WINDOW_H / HEIGHT * w - text_size[0] / 2 + WINDOW_X), int(WINDOW_H / HEIGHT * h))
                rectangle = np.ones((text_size[1] + 10, text_size[0] + 10, 3), np.uint8) * 255
                alpha = 0.5
                try:
                    depth3[text_pos[1] - 5 : text_pos[1] + text_size[1] + 5, text_pos[0] - 5 : text_pos[0] + text_size[0] + 5] = \
                        cv2.addWeighted(depth3[text_pos[1] - 5 : text_pos[1] + text_size[1] + 5, text_pos[0] - 5 : text_pos[0] + text_size[0] + 5],
                                        alpha, rectangle, 1 - alpha, 0)
                    depth3 = cv2.putText(depth3, text,
                                        (text_pos[0], text_pos[1] + text_size[1]), font,
                                        font_scale, color, thickness, cv2.LINE_AA)
                    t_w = int(HEIGHT / WINDOW_H * text_size[0])
                    t_h = int(HEIGHT / WINDOW_H * text_size[1])
                    find[h - t_h - 1 : h + 1, w - t_w - 1: w + t_w + 2] = 0
                except:
                    pass

    cv2.imwrite(f"{ROOT_FOLDER}/video_utils/videos/{OUTPUT_FOLDER}/{filename.rsplit(os.sep, 1)[1]}", depth3)