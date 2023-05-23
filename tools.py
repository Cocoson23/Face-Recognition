import cv2
import os
import numpy

""" 从path路径读取图像并检测出人脸，将人脸转换至image_size大小 """
def read_images(path, image_size):
    names = []
    training_images, training_labels = [], []
    label = 0
    for dirname, subdirnames, filenames in os.walk(path):
        for subdirname in subdirnames:
            names.append(subdirname)
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                img = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, image_size)
                training_images.append(img)
                training_labels.append(label)
            label += 1
    training_images = numpy.asarray(training_images, numpy.uint8)
    training_labels = numpy.asarray(training_labels, numpy.int32)

    """ return 数据库中人名数组、所有转换完毕的人脸图像以及对应的标签 """
    return names, training_images, training_labels


def reName(root):
    """ 将root路径下的图片全部重命名为0开始依次+1.jpg """
    files = os.listdir(root)
    i = 0
    for file in files:
        new_filename = os.path.join(root, str(i)+".pgm")
        os.rename(os.path.join(root, file), os.path.join(root, new_filename))
        i += 1


def imageConvert(input_folder, output_folder):
    """ 检测原始图片中的人脸部分，并resize成200*200的图片 """
    # 若output_folder文件夹不存在则自动创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 人脸检测级联
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

    images = os.listdir(input_folder)
    count = 0
    for image_path in images:
        image = cv2.imread(os.path.join(input_folder, image_path), cv2.IMREAD_GRAYSCALE)
        faces = face_cascade.detectMultiScale(image, 1.3, 5, minSize=(120, 120))
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_img = cv2.resize(image[y:y + h, x:x + w], (200, 200))
            face_filename = '%s/%d.pgm' % (output_folder, count)
            count += 1
            cv2.imwrite(face_filename, face_img)
