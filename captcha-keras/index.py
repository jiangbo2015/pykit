from captcha.image import ImageCaptcha
import matplotlib
from keras.utils.np_utils import to_categorical
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
import string
import sys

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'


characters = string.digits + string.ascii_uppercase

width, height, n_len, n_class = 170, 80, 4, len(characters)

# generator = ImageCaptcha(width=width, height=height)
# random_str = ''.join([random.choice(characters) for j in range(4)])
# img = generator.generate_image(random_str)

# plt.imshow(img)
# plt.title(random_str)
# plt.axis("off")
# plt.show()




def gen(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)

    while True:
        for i in range(batch_size):

            # 产生随机字母和数字
            random_str = ''.join([random.choice(characters) for j in range(4)])

            X[i] = generator.generate_image(random_str)

            for j, ch in enumerate(random_str):
                # print(characters.find(ch))
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
                print(y)
                print('-----')
        yield X, y


def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

X, y = next(gen(1))
# plt.imshow(X[0])
# plt.title(decode(y))
# plt.axis("off")
# plt.show()

sys.exit()

from keras.models import *
from keras.layers import *

input_tensor = Input((height, width, 3))

x = input_tensor
for i in range(4):
    x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.25)(x)
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]
model = Model(input=input_tensor, output=x)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])



model.fit_generator(gen(), samples_per_epoch=20, nb_epoch=1,
                    validation_data=gen(), nb_val_samples=10)




X, y = next(gen(1))
y_pred = model.predict(X)

print('real: %s\npred:%s'%(decode(y), decode(y_pred)))
# plt.title('real: %s\npred:%s'%(decode(y), decode(y_pred)))
# plt.imshow(X[0], cmap='gray')
# plt.axis('off')



# from tqdm import tqdm
# def evaluate(model, batch_num=20):
#     batch_acc = 0
#     generator = gen()
#     for i in tqdm(range(batch_num)):
#         X, y = generator.next()
#         y_pred = model.predict(X)
#         batch_acc += np.mean(map(np.array_equal, np.argmax(y, axis=2).T, np.argmax(y_pred, axis=2).T))
#     return batch_acc / batch_num

# evaluate(model)







