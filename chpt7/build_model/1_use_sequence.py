# -*- coding: utf-8 -*-
__author__ = 'Mike'
import keras
from keras import layers

model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model_1 = keras.Sequential()
model_1.add(layers.InputLayer(input_shape=(3,))) #注意这里是InputLayer
model_1.add(layers.Dense(64, activation="relu"))
model_1.add(layers.Dense(10, activation="softmax"))

#这种方法只能通过调用模型来完成构建
# model_1.build(input_shape=(None, 3))
print(model_1.weights)
print(model_1.summary())#查看模型概览