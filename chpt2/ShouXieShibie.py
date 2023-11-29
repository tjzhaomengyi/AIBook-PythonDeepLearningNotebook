# -*- coding: utf-8 -*-
__author__ = 'Mike'
import tensorflow as tf
import math
from tensorflow import keras;
# import keras;
from keras.datasets import mnist
import numpy as np
tf.enable_eager_execution()
print(keras.__version__)
print(tf.__version__)
#创建模型的w和b变量
class NaiveDense:
        def __init__(self, input_size, output_size, activation):
            self.activation = activation
            #todo:这列的w_shape为什么是input_size和output_size，表示有多少个输入（列）那么对应的多少个（行）的输出，所以矩阵参数的初始化是：
            #   input_size列,
            w_shape = (input_size, output_size) #创建一个形状为(input_size, output_size)的矩阵w，并将其随机初始化
            w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
            self.W = tf.Variable(w_initial_value)

            b_shape = (output_size,)
            b_initial_value = tf.zeros(b_shape) #创建一个形状为（output)size,）的零向量b
            self.b = tf.Variable(b_initial_value)

        #前向传播
        def __call__(self, inputs):
            return self.activation(tf.matmul(inputs, self.W) + self.b)

        #获取权重
        @property
        def weights(self):
            return [self.W, self.b]




class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights


class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
        assert len(images) == len(labels)
        self.index = 0;
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)

    def next(self):
        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels

#做一轮训练
def one_training_step(model, images_batch, labels_batch):
    with tf.GradientTape() as tape: #前向计算
        predictions = model(images_batch)
        per_sample_losses = keras.losses.sparse_categorical_crossentropy(labels_batch, predictions)
        average_loss = tf.reduce_mean(per_sample_losses)
        #计算损失相对于权重的梯度。输出的gradients是一个列表，每个元素对应model.weights列表中的权重
    gradients = tape.gradient(average_loss, model.weights)
    optimizer.apply_gradients(zip(gradients, weights)) #这里必须使用from tensorflow import keras才行
    # update_weights_keras(gradients, model.weights)
    return average_loss

learning_rate = 1e-3

def update_weights(gradients, weigths):
    for g, w in zip(gradients, weigths):
        w.assign_sub(g * learning_rate) #对w重新赋值，g梯度乘以学习率

#但是在keras中我们直接使用optimizer来更新权重
# optimizer = tf.keras.optimizers.AdamOptimizer(lr=1e-3)
# optimizer = tf.compat.v1.train.AdamOptimizer(1e-3)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
optimizer = keras.optimizers.SGD(1e-3) #注意：如果使用apply_gradients()一定要使用tensorflow.keras.opt...

def update_weights_keras(gradients, weights):
    optimizer.apply_gradients(zip(gradients, weights))

#完整的训练
def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter}")
        batch_generator = BatchGenerator(images, labels)
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
            loss = one_training_step(model, images_batch, labels_batch)
            if batch_counter == 0:
                print(f"loss at batch {batch_counter} : {loss:.2f}")




(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

model = NaiveSequential([NaiveDense(input_size= 28 * 28, output_size=512, activation=tf.nn.relu),
                         NaiveDense(input_size= 512, output_size=10, activation=tf.nn.softmax)])
assert len(model.weights) == 4
#创建模型
fit(model, train_images, train_labels, epochs=10, batch_size=128)

predictions = model(test_images)
predictions = predictions.numpy()
predicted_labels = np.argmax(predictions, axis=1)
matches = predicted_labels == test_labels
print(f"acc:{matches.mean():.2f}")