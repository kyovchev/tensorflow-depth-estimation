import cv2
import glob

FRAMES_FOLDER = "video_2_cut_depth"
FRAMES_PER_SECOND = 10

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

ROOT_FOLDER = "D:/tensorflow-depth-estimation"

frames = []
for filename in glob.glob(f"{ROOT_FOLDER}/video_utils/videos/{FRAMES_FOLDER}/*.jpg"):
    print(filename)
    frame = cv2.imread(filename)

    # draw the processing target 928x696 rectangle
    frame = cv2.rectangle(frame, (WINDOW_X, WINDOW_Y), (WINDOW_X + WINDOW_W, WINDOW_Y + WINDOW_H),
                          (0, 0, 255), 2)

    #cv2.imshow("frame", frame)
    #cv2.waitKey(0)
    #break
    frames.append(frame)

h, w, _ = frames[0].shape
out = cv2.VideoWriter(f"{ROOT_FOLDER}/video_utils/videos/{FRAMES_FOLDER}_out.avi",
                      cv2.VideoWriter_fourcc(*'DIVX'), FRAMES_PER_SECOND, (w, h))

for i in range(len(frames)):
    out.write(frames[i])
out.release()