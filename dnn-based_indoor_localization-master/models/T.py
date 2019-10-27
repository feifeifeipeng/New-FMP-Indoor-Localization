from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D

np.random.seed(10)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 将features转换为三维矩阵
x_train3D = x_train.reshape(x_train.shape[0], 784, 1).astype('float32')
x_test3D = x_test.reshape(x_test.shape[0], 784, 1).astype('float32')
# 将feature标准化
x_train3D_normalize = x_train3D / 255
x_test3D_normalize = x_test3D / 255
# 将label进行one hot转换
y_trainOneHot = np_utils.to_categorical(y_train)
y_testOneHot = np_utils.to_categorical(y_test)

# 建立一个Sequential线性堆叠模型
model = Sequential()
model.add(Conv1D(filters=16,
                 kernel_size=25,
                 padding='same',
                 input_shape=(784, 1),
                 activation='relu'))
model.add(MaxPooling1D(pool_size=4))
# 建立卷积层2
model.add(Conv1D(filters=10,
                 kernel_size=25,
                 padding='same',
                 activation='relu'))
# 建立池化层2
model.add(MaxPooling1D(pool_size=4))
model.add(Dropout(0.25))
# 建立平坦层
model.add(Flatten())
# 建立隐蔽层
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
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
