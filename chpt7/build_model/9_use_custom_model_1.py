# -*- coding: utf-8 -*-
__author__ = 'Mike'
import keras
from keras import layers
import tensorflow as tf
from keras.datasets import mnist

loss_fn = keras.losses.SparseCategoricalCrossentropy()
loss_tracker = keras.metrics.Mean(name="loss")
class CutomModel(keras.Model):
    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            predictions = self(inputs, trainable=True) #注意这里调用的model(x)的方法，不是调用predict，因为predict无法记录梯度
            loss = loss_fn(targets, predictions)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        loss_tracker.update(loss) #这里只更新loss指标
        return {"loss": loss_tracker.result()}

    @property
    def metrics(self): #这里列出需要再不同轮次之间进行重置的指标
        return [loss_tracker]

#数据
(images, labels), (test_images, test_labels) = mnist.load_data()
print(labels[:10])
images = images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]

inputs = keras.Input(shape=(28 * 28,))
features = layers.Dense(512, activation="relu")(inputs)
features = layers.Dropout(0.5)(features)
outputs = layers.Dense(10, activation="softmax")(features)
model = CutomModel(inputs, outputs)


model.compile(optimizer=keras.optimizers.RMSprop(), loss=loss_fn) #,传入优化器即可，损失在模型之外定义
model.fit(train_images, train_labels, epochs=3)