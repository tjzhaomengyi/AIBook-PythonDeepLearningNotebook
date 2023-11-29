# -*- coding: utf-8 -*-
__author__ = 'Mike'
'''
这个他妈了个逼的tensorflow的内部导入在pycharm中不出现代码提示，必须使用keras
tensorflow=1.15 对应 keras=2.3.1，tf和keras都对应这个版本的最高版本    
'''
# import tensorflow.keras as keras
# from tensorflow.python import keras
import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras import models
import keras
from keras import Sequential
from keras import models
from keras import layers

print(tf.__version__)
print(keras.__version__)


class SimpleDense(keras.layers.Layer):

    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(shape=(input_dim, self.units), initializer="random_normal")
        self.b = self.add_weight(shape=(self.units,), initializer="zeros")

    def call(self, inputs):
        y = tf.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y

my_dense = SimpleDense(units=32, activation=tf.nn.relu) #有32个输出单元的密集层
input_tensor = tf.ones(shape=(2, 784))
output_tensor = my_dense(input_tensor)
print(output_tensor)
#使用Keras代替上面的写法，更简单
model = models.Sequential([
    layers.Dense(32, activation="relu"),
    layers.Dense(3)
])
print(model)

#线性分类器
model = keras.Sequential([keras.layers.Dense(1)])
model.compile(optimizer="rmsprop", loss="mean_squared_error", metrics=["accuracy"])
#或者写为
model.compile(optimizer=keras.optimizers.RMSprop(), loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.BinaryAccuracy()])
