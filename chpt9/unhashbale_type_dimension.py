# -*- coding: utf-8 -*-
__author__ = 'Mike'
import tensorflow as tf
import keras

dim_2048 = tf.Dimension(2048)
# 假设 input_tensor 是形状为 (?, 2048) 的张量
input_tensor = tf.placeholder(tf.float32, shape=(None, dim_2048))
print(type(input_tensor))
print(input_tensor.get_shape())
print(input_tensor)
# 设置输入张量的形状
# input_tensor.set_shape((None, 2048))
model = keras.applications.xception.Xception(weights="imagenet", include_top=True)


classifier_input = keras.Input(shape=(10, 10, 2048))
x = classifier_input
x = model.get_layer("avg_pool")(x)
print(x)
pre_layer = model.get_layer("predictions")
x1 = pre_layer(x)
x = keras.layers.Dense(1000, activation='softmax')(input_tensor)