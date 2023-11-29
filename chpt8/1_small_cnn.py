# -*- coding: utf-8 -*-
__author__ = 'Mike'
import keras
from keras import layers
from keras.datasets import mnist

inputs = keras.Input(shape=(28, 28, 1)) #输入图片大小28 * 28 * 1
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs) #kernel_size卷积核大小3*3，filter表示在纵轴通道少有多少个滤波器
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPool2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x) #把3维的数据拍成1维
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)) #这列每张图片是一个三维数据
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype("float") / 255
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.3f}")
