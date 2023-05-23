import cv2

from tools import read_images

# 创建人脸检测级联
face_cascade = cv2.CascadeClassifier(r"./cascades/haarcascade_frontalface_default.xml")

# 训练数据集路径
path_to_training_images = './datasets/face'
# 测试数据集路径
path_to_test_images = './datasets/test'
# 测试结果数据路径
path_output = './test_output'
# 训练时设置的人脸图像大小
training_image_size = (200, 200)
# 读取人名、训练图像以及标签
names, training_images, training_labels = read_images(path_to_training_images, training_image_size)

model = cv2.face.EigenFaceRecognizer_create()
model.read('./model.xml')

num = 0

while True:
    # 读取测试数据集每一帧
    img_read_path = path_to_test_images +'/' + str(num) +'.jpg'
    img_save_path = path_output + '/' + str(num) + '.jpg'
    frame = cv2.imread(img_read_path)
    # 读取图像成功执行下列代码，否则退出
    if frame is not None:
        # 对每一帧检测人脸
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        # 检测结果若有人脸则将其裁剪出来放入模型进行检测
        # 若无人脸则跳过
        for (x, y, w, h) in faces:
            # 画人脸框
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 4)
            roi = frame[x:x+w, y:y+h]
            if roi.size == 0:
                continue
            # 调整检测人脸至训练时人脸图像大小
            roi = cv2.resize(roi, training_image_size)
            # 转换至灰色图像
            roi = cv2.cvtColor(roi, cv2.COLOR_RGBA2GRAY)
            # 检测结果
            label, confidence = model.predict(roi)
            # 在人脸框左上角标注人名及置信度
            text = '%s, confidence=%.2f' % (names[label], confidence)
            cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        num = num + 1
        cv2.imwrite(img_save_path, frame)
    else:
        break
