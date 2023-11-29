# -*- coding: utf-8 -*-
__author__ = 'Mike'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras


# tf.enable_eager_execution()

num_samples_per_class = 1000
#协方差矩阵conv，对应一个从左下方到右上方的椭圆形点云
negative_samples = np.random.multivariate_normal(mean=[0, 3], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class)
positive_samples = np.random.multivariate_normal(mean=[3, 0], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class)

#将两个类别堆叠成一个形状为(2000,2)的数组
inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
#生成正负例样本的label
targets = np.vstack((np.zeros((num_samples_per_class, 1),dtype="float32"), np.ones((num_samples_per_class,1),dtype="float32")))

plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.show()

#创建线性分类器
input_dim = 2
output_dim = 1 #样本属于0就接近0，属于1就接近1
W=tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim))) #确认输入维度，input_dim维度，然后得到output_dim的结果
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,))) #结果是一个一维张量，因为是二分类，所以输出维度不写就是1维度，接近0就是负例，接近1就是正例
print(W)
print(b)

def model(inputs):
    return tf.matmul(inputs, W) + b

def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    print(per_sample_losses)
    return tf.reduce_mean(per_sample_losses)

learning_rate = 0.1
def training_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(targets, predictions)
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    W.assign_sub(grad_loss_wrt_W * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    return loss

for step in range(40):
    loss = training_step(inputs, targets)
    print(f"Loss at step {step}: {loss:.4f}")

predictions = model(inputs)
x = np.linspace(-1, 4, 100)
y = -W[0] / W[1] * x + (0.5 - b) / W[1]
plt.plot(x, y, "-r")
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()



