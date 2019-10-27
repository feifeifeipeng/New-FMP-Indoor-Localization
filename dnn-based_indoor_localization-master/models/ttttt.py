import numpy as np
import keras
from  keras.layers import Conv2D,MaxPool2D,Flatten,Dense
seed=13
np.random.seed(seed)
data_size = [10, 10]
data_2d = np.random.normal(size=data_size)
data_2d = np.expand_dims(data_2d, 0)
data_2d = np.expand_dims(data_2d, 3)
print(data_2d.shape)

# 定义卷积层
conv_size = 2
conv_stride_size = 2
convolution_2d_layer = keras.layers.Conv2D(filters=1, kernel_size=(conv_size, conv_size), strides=(conv_stride_size, conv_stride_size), input_shape=(data_size[0], data_size[0], 1))
# convolution_2d_layer = keras.layers.Conv2D(filter=1, kernel_size=kernel, strides=[1,1], padding="valid", activation="relu", name="convolution_2d_layer", input_shape=(1, data_size[0], data_size[0]))


# 定义最大化池化层
pooling_size = (2, 2)
max_pooling_2d_layer = keras.layers.MaxPool2D(pool_size=pooling_size, strides=1, padding="valid", name="max_pooling_2d_layer")

# 平铺层，调整维度适应全链接层
reshape_layer = keras.layers.core.Flatten(name="reshape_layer")

# 定义全链接层
full_connect_layer = keras.layers.Dense(5, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=seed), bias_initializer="random_normal", use_bias=True, name="full_connect_layer")

model_2d = keras.Sequential()
model_2d.add(convolution_2d_layer)
model_2d.add(max_pooling_2d_layer)
model_2d.add(reshape_layer)
model_2d.add(full_connect_layer)

# 打印 full_connect_layer 层的输出
output = keras.Model(inputs=model_2d.input, outputs=model_2d.get_layer('full_connect_layer').output).predict(data_2d)
print("======================卷积结果=========================")
print(output)

# 打印网络结构
print("======================网络结构=========================")
print(model_2d.summary())