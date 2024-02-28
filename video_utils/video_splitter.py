import cv2

FILENAME = "video_1"
FORMAT = "mp4"

capture = cv2.VideoCapture(f"./video_utils/videos/{FILENAME}.{FORMAT}")

frame_number = 0

while True:
    success, frame = capture.read()

    if success:
        frame = cv2.resize(frame, (1920, 1080), interpolation = cv2.INTER_AREA)
        cv2.imwrite(f"./video_utils/videos/{FILENAME}/frame_{frame_number:05d}.jpg", frame)

    else:
        break

    frame_number = frame_number + 1

capture.release()