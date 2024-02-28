import cv2
import glob

FRAMES_FOLDER = "video_1_cut_kitti_unet_depth_object"
FRAMES_PER_SECOND = 2

ROOT_FOLDER = "D:/tensorflow-depth-estimation"

frames = []
for filename in glob.glob(f"{ROOT_FOLDER}/video_utils/videos/{FRAMES_FOLDER}/*.jpg"):
    print(filename)
    frame = cv2.imread(filename)

    # draw the processing target 928x696 rectangle
    frame = cv2.rectangle(frame, ((1920 - 928) // 2, 0),
                          ((1920 + 928) // 2, 696), (0, 0, 255), 2)

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