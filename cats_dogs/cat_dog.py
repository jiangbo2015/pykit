

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing.image import img_to_array, load_img
import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import matplotlib
import matplotlib.pyplot as plt


# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 15
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

# 生产model
model = create_model()

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')



# 训练
def train():
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    model.save_weights('second_try.h5')

# 评估模型
def evaluate_generator():
    model.load_weights('first_try.h5')
    steps = validation_generator.n//validation_generator.batch_size
    e = model.evaluate_generator(
        validation_generator, 
        steps=steps # steps可以省略，评估结果一样
        )
    print(e) # [0.5820711821317672, 0.7375] 损失/精度
    

def predict_generator():
    model.load_weights('first_try.h5')
    # 使用测试集合
    test_generator = test_datagen.flow_from_directory(
        './test/',
        target_size=(img_width, img_height), 
        batch_size=batch_size,
        class_mode='binary',
        seed=42,
        shuffle=False)
    test_generator.reset()
    steps = test_generator.n//test_generator.batch_size
    r = model.predict_generator(test_generator, steps = steps)
    print(r.shape)    # (32, 1)
    print(test_generator.class_indices) # {'cats': 0, 'dogs': 1}
    print([x[0] for x in r]) # [0.4,0.3,...,0.9] 预测结果概率值

def predict():
    model.load_weights('first_try.h5')
    img = './data/validation/cats/cat.1009.jpg'
    # img = './data/validation/dogs/dog.11005.jpg'
    img = load_img(img,False,target_size=(img_width,img_height))
    x = img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x) #返回概率
    print(preds) #[[0.48104638]]

def predict_classes():
    model.load_weights('first_try.h5')
    img = './data/validation/cats/cat.1009.jpg'
    img = load_img(img,False,target_size=(img_width,img_height))
    x = img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict_classes(x) #返回classes类别
    print(preds) #[[0]]

def predict_and_show():
    model.load_weights('first_try.h5')
    img_array1 = ['./data/validation/dogs/dog.11005.jpg','./data/validation/dogs/dog.11008.jpg', './data/validation/cats/cat.1009.jpg']
    img_array = ['./data/validation/cats/cat.1009.jpg']
    img_text = []
    for i in img_array:
        img = load_img(i,False,target_size=(img_width,img_height))
        x = img_to_array(img)
        x = x/255.0
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x) #返回概率
        
        print(preds.shape)
        
        pred = model.predict_classes(x) #返回所属类别
        print(preds, pred)
        print(preds.shape)
    #     print(preds)
    #     if preds[0][0] == 1:
    #         text = 'dog'
    #     else:
    #         text = 'cats'
    #     img_text.append(text)
    # print(img_text)

    # imgs = []
    # for i,x in enumerate(img_array):
    #     image = Image.open(x)
    #     draw = ImageDraw.Draw(image)
    #     font = ImageFont.truetype('/Users/jiangbo/Library/Fonts/Arial.ttf', 60)
    #     draw.text((80,80), img_text[i], font=font)
    #     imgs.append(image)
    # fig, ax = plt.subplots(1, len(img_text))
    # for i, x in enumerate(img_text):
    #     ax[i].imshow(imgs[i])
    # plt.show()

# predict_and_show()

# train()

# evaluate()
# evaluate_generator()

# predict_generator()
predict()
# predict_classes()




