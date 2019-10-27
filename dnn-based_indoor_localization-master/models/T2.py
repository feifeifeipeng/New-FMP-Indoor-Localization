import numpy as np

from keras.datasets import mnist

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
np.random.seed(10)

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# data pre-processing
x_train3D = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test3D = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
# 将feature标准化
x_train3D_normalize = x_train3D / 255
x_test3D_normalize = x_test3D / 255
# 将label进行one hot转换
y_trainOneHot = np_utils.to_categorical(y_train)
y_testOneHot = np_utils.to_categorical(y_test)

# Another way to build your CNN
model = Sequential()

# 卷积层  Conv layer 1 output shape (32, 28, 28)
model.add(Convolution2D(
    input_shape=(28, 28, 1),
    filters=16,  # 滤波器数量
    kernel_size=5,  # 滤波器大小5x5
    strides=1,  # 步长1
    padding='same',  # Padding method
    data_format='channels_first',
))
model.add(Activation('relu'))

# 池化 Pooling layer 1 (max pooling) output shape (32, 14, 14)
model.add(MaxPooling2D(
    pool_size=2,  # 卷积核大小
    strides=2,  # 卷积步长
    padding='same',  # Padding method
    data_format='channels_first',
))

# Conv layer 2 output shape (64, 14, 14)
model.add(Convolution2D(32, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (64, 7, 7)
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
model.add(Flatten())  # 将数据抹平为一维
model.add(Dense(128))  # 接入全链接层
model.add(Activation('relu'))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))

# Another way to define your optimizer
adam = Adam(lr=1e-4)
print(model.summary())

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
train_history = model.fit(x=x_train3D_normalize,
                          y=y_trainOneHot, validation_split=0.2,
                          epochs=30, batch_size=300, verbose=2)
import matplotlib.pyplot as plt


def show_train_history(train_history, train, validation):
    """
    显示训练过程

    参数：
        train_history - 训练结果存储的参数位置
        train - 训练数据的执行结果
        validation - 验证数据的执行结果

"""
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')
scores = model.evaluate(x_test3D_normalize, y_testOneHot)
scores[1]

