# -*- coding: utf-8 -*-
__author__ = 'Mike'
import tensorflow as tf
import numpy as np

# 创建一个 TensorFlow Tensor 对象
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
print(type(tensor))
# 创建 TensorFlow Session
with tf.Session() as sess:
    # 使用 run 方法运行 Session 获取 NumPy 数组
    numpy_array = sess.run(tensor)

# 现在 numpy_array 是一个 NumPy 数组
print(numpy_array)

img_width = 200
img_height = 200
iterations = 30  # 梯度上升步数
learning_rate = 10.
image = tf.random.uniform(
    minval=0.4,
    maxval=0.6,
    shape=(1, img_width, img_height, 3))
print(image)
with tf.Session() as sess:
    img_arr = sess.run(image)
print(img_arr)