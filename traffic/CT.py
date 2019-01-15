from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout

from keras import backend as K
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from imutils import paths
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import random
import cv2
import os
import sys

"""
定义的常量
"""
EPOCHS = 30 # 循环次数
BATCH_SIZE = 32 #每批处理的数量
CLASS_NUM = 62 # 分类的数量
IMG_SIZE = 32 # 图片大小


"""
数据加载，返回图片列表，分类名称（如1，2等）
该数据加载方法不太好，后面优化
"""
def load_data(path):
    images = []
    labels = []
    imagePaths = sorted(list(paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagePaths)

    for file in imagePaths:
        image = cv2.imread(file)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        image = img_to_array(image)
        images.append(image)

        label = int(file.split(os.path.sep)[-2])   
        labels.append(label)
    
    images = np.asarray(images, dtype="float") / 255.0
    labels = np.array(labels)

    labels = to_categorical(labels, num_classes=CLASS_NUM) # one-hot编码                      
    return images, labels


def gendata_from_file(filepath, batch_size):
    pass



"""
创建模型
"""
def create_model():
    
    # 根据backend 确定输入
    inputShape = (IMG_SIZE, IMG_SIZE, 3)
    if K.image_data_format() == "channels_first":
        inputShape = (3, IMG_SIZE, IMG_SIZE)

    model = Sequential()

    # 第一层卷集核数量， 卷积核大小，填充方式，输入
    model.add(Conv2D(32, kernel_size=(5, 5), input_shape=inputShape))
    model.add(Activation("relu")) #激活
    model.add(MaxPooling2D(pool_size=(2, 2))) #池化，使用MaxPolling
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(500, activation="relu")) #全链接层
    # model.add(Dropout(0.5))

    model.add(Dense(CLASS_NUM, activation="softmax")) #最后输出CLASS_NUM个

    return model


"""
训练模型，先加载数据，使用图像增强来增加数据集数量，
"""
def train():
    x, y = load_data('./data/train/')
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    # x_test, y_test = load_data('./data/test/')

    datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
            height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
            horizontal_flip=True, fill_mode="nearest") 

    model = create_model()

    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer='rmsprop',
        metrics=["accuracy"])

    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(x_test, y_test), steps_per_epoch=len(x_train) // BATCH_SIZE,
        epochs=EPOCHS, verbose=1)

    model.save('traffic_with_dropout.h5')

    #画图
    x = np.arange(0, EPOCHS)
    y = history.history['val_loss']
    y1 = history.history['val_acc']
    y2 = history.history['loss']
    y3 = history.history['acc']

    plt.figure()
    plt.plot(x, y, label="val_loss")
    plt.plot(x, y1, label="val_acc")
    plt.plot(x, y2, label="loss")
    plt.plot(x, y3, label="acc")
    plt.legend()
    plt.savefig('traffic_with_dropout.png')
    plt.show()
    

    
"""
预测
"""
def predict():
    model = load_model('traffic.h5')
    img = cv2.imread('./data/test/00031/00124_00000.png')
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict_classes(x)
    print(preds) # [31]

        
if __name__=='__main__':
    train()
    # predict()



# 32, 64, 3,3
# loss: 0.3643 - acc: 0.8854 - val_loss: 0.2460 - val_acc: 0.9274
# 
# 32, 64, 5,5
# loss: 0.3329 - acc: 0.8952 - val_loss: 0.1920 - val_acc: 0.9488
# 
# 0.3537 - acc: 0.8919 - val_loss: 0.2467 - val_acc: 0.9306
# 
# loss: 0.2094 - acc: 0.9427 - val_loss: 0.1511 - val_acc: 0.9607









