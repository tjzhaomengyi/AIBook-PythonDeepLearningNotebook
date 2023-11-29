# -*- coding: utf-8 -*-
__author__ = 'Mike'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras


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


#使用keras验证数据集升级训练
model = keras.Sequential([keras.layers.Dense(1)]) #
model.add(keras.layers.Dense(2, activation='softmax'))
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.1), loss=keras.losses.MeanSquaredError(),metrics=[keras.metrics.BinaryAccuracy()])
indices_permutation = np.random.permutation(len(inputs)) #对输入进行一个随机排序，然后给shuffle
shuffle_inputs = inputs[indices_permutation] #
shuffle_targets = inputs[indices_permutation]
#生成验证集
num_validation_samples = int(0.3 * len(inputs))
val_inputs = shuffle_inputs[:num_validation_samples]
val_targets = shuffle_targets[:num_validation_samples]

training_inputs = shuffle_inputs[num_validation_samples:]
training_targets = shuffle_targets[num_validation_samples:]
model.fit(x=training_inputs, y=training_targets, epochs=5, batch_size=16, validation_data=(val_inputs, val_targets))
loss_and_metrics = model.evaluate(val_inputs, val_targets, batch_size=128)
predictions = model.predict(val_inputs, batch_size=128)
print(predictions[0:10])