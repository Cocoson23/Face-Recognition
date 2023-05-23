import cv2

from tools import read_images

# 数据集根路径
path_to_training_images = './datasets/face'
# 训练时设置的人脸图像大小
training_image_size = (200, 200)
# 读取人名、训练图像以及标签
names, training_images, training_labels = read_images(path_to_training_images, training_image_size)
# 创建人脸检测模型并训练
model = cv2.face.EigenFaceRecognizer_create()
model.train(training_images, training_labels)
# 保存训练好的模型到当前路径下
model.save('./model.xml')


