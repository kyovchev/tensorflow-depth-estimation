import glob
import numpy as np
import tensorflow as tf
import cv2
import imutils
import os

FRAMES_FOLDER = "video_2_cut"
OUTPUT_FOLDER = "video_2_cut_depth"

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

    depth3 = image.copy()

    heatmap = cv2.applyColorMap((255 - (prediction - ADDED_D) / MAX_D * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heatmap2 = cv2.resize(heatmap, (WINDOW_W, WINDOW_H))
    alpha = 0.1
    depth3[int(WINDOW_Y + HEATMAP_Y / HEIGHT * WINDOW_H) : WINDOW_Y + WINDOW_H,
           WINDOW_X : WINDOW_X + WINDOW_W] = \
        cv2.addWeighted(depth3[int(WINDOW_Y + HEATMAP_Y / HEIGHT * WINDOW_H) : WINDOW_Y + WINDOW_H,
                               WINDOW_X : WINDOW_X + WINDOW_W],
                        alpha, heatmap2[int(HEATMAP_Y / HEIGHT * WINDOW_H) :, :], 1 - alpha, 0)

    points = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    if USE_EMPTY_ROAD_IMAGE:
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

        for i in range(1, ret1):
            component = np.zeros((HEIGHT, WIDTH), np.uint8)
            component[lab == i] = 255
            if np.count_nonzero(component) > 100:
                rect = cv2.boundingRect(component)
                points[rect[1] + rect[3], rect[0] + rect[2] // 2] = 255
    else:
        points[HEIGHT // 2     , WIDTH // 2 - 100] = 255
        points[HEIGHT // 2     , WIDTH // 2 - 45 ] = 255
        points[HEIGHT // 2     , WIDTH // 2 + 10 ] = 255
        points[HEIGHT // 2 + 30, WIDTH // 2 - 100] = 255
        points[HEIGHT // 2 + 30, WIDTH // 2 - 45 ] = 255
        points[HEIGHT // 2 + 30, WIDTH // 2 + 10 ] = 255
        points[HEIGHT // 2 + 60, WIDTH // 2 - 100] = 255
        points[HEIGHT // 2 + 60, WIDTH // 2 - 45 ] = 255
        points[HEIGHT // 2 + 60, WIDTH // 2 + 10 ] = 255
        points[HEIGHT // 2 + 90, WIDTH // 2 - 100] = 255
        points[HEIGHT // 2 + 90, WIDTH // 2 - 45 ] = 255
        points[HEIGHT // 2 + 90, WIDTH // 2 + 10 ] = 255


    font = cv2.FONT_HERSHEY_SIMPLEX 
    font_scale = 0.7
    color = (255, 0, 0) 
    thickness = 1

    find = points.copy()

    for h in range(220, 60, -1):
        for w in range(WIDTH):
            if find[h, w] > 128:
                dist = prediction[h, w, 0]
                text = f"{dist:.2f} m"
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_pos = (int(WINDOW_X + WINDOW_H / HEIGHT * w - text_size[0] / 2), int(WINDOW_Y + WINDOW_H / HEIGHT * h))
                rectangle = np.ones((text_size[1] + 10, text_size[0] + 10, 3), np.uint8) * 255
                alpha = 0.5
                try:
                    #depth3[text_pos[1] - 5 : text_pos[1] + text_size[1] + 5, text_pos[0] - 5 : text_pos[0] + text_size[0] + 5] = \
                    #    cv2.addWeighted(depth3[text_pos[1] - 5 : text_pos[1] + text_size[1] + 5, text_pos[0] - 5 : text_pos[0] + text_size[0] + 5],
                    #                    alpha, rectangle, 1 - alpha, 0)
                    depth3 = cv2.putText(depth3, text,
                                         (text_pos[0], text_pos[1] + text_size[1]), font,
                                         font_scale, color, thickness, cv2.LINE_AA)
                    t_w = int(HEIGHT / WINDOW_H * text_size[0])
                    t_h = int(HEIGHT / WINDOW_H * text_size[1])
                    find[h - t_h - 1 : h + 1, w - t_w - 1: w + t_w + 2] = 0
                except:
                    pass

    cv2.imwrite(f"{ROOT_FOLDER}/video_utils/videos/{OUTPUT_FOLDER}/{filename.rsplit(os.sep, 1)[1]}", depth3)