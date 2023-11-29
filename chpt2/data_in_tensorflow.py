# -*- coding: utf-8 -*-
__author__ = 'Mike'
from keras.datasets import mnist
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
print(type(train_images))
print(train_images.ndim) #张量维度
print(train_images.shape) #张量维度描述
print(train_images.dtype) #张量每个数据的数据类型

digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
print(train_labels[4])

#张量操作
my_slice = train_images[10:100] #张量的切片
print(my_slice.shape)


#批量操作
batch = train_images[:128]

#张量变形
x = np.array([[0, 1],
             [2, 3],
             [4, 5]])
print(x.shape)
x = x.reshape([6, 1])
print(x)
print(np.transpose(x)) #转置矩阵

'''
向量解释
'''