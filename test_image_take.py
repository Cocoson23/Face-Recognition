import cv2
import os
""" 数据集采集 """
""" 调用电脑自带摄像头获取测试并存放到output_folder路径 """

output_folder = './datasets/test'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

camera = cv2.VideoCapture(0)
count = 0
while (cv2.waitKey(1) == -1):
    success, frame = camera.read()
    if success:
        frame_path = '%s/%d.jpg' % (output_folder, count)
        cv2.imwrite(frame_path, frame)
        count += 1
        cv2.imshow('Capturing Faces...', frame)

camera.release()
cv2.destroyAllWindows()
