# -*- coding: utf-8 -*-
__author__ = 'Mike'
from keras.datasets import mnist
import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt
from keras import regularizers


def get_model():
    model = keras.Sequential([
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def get_model_L2():
    model = keras.Sequential([
        layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.02)),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def get_model_dropout():
    model = keras.Sequential([
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

(train_images, train_labels), _ = mnist.load_data()
train_images = train_images.reshape(60000, 28 * 28)
train_images = train_images.astype("float32") / 255
#在原来的每张图片上加上噪声和全0通道
train_images_with_noise_channels = np.concatenate(
    [train_images, np.random.random((len(train_images), 28 * 28))], axis=1
)
train_images_with_zeros_channels = np.concatenate(
    [train_images, np.zeros((len(train_images), 784))], axis=1
)
'''
model_noise = get_model()
history_noise = model_noise.fit(train_images_with_noise_channels, train_labels, epochs=10, batch_size=128, validation_split=0.2)

model_zero = get_model()
history_zero = model_zero.fit(train_images_with_zeros_channels, train_labels, epochs=10, batch_size=128, validation_split=0.2)

val_acc_noise = history_noise.history["val_accuracy"]
val_acc_zeros = history_zero.history["val_accuracy"]
epochs = range(1, 11)
plt.plot(epochs, val_acc_noise, "b-", label="Validation accuracy with noise channel")
plt.plot(epochs, val_acc_zeros, "b--", label="Validation accuracy with zeros channel")
plt.title("白噪声和零通道对准确率的影响")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
'''
#L2正则验证
model_l2 = get_model_L2()
history_l2 = model_l2.fit(train_images, train_labels, epochs=10, batch_size=128, validation_split=0.2)
model = get_model()
history = model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_split=0.2)

val_acc = history.history["val_loss"]
val_acc_l2 = history_l2.history["val_loss"]
epochs = range(1, 11)
plt.plot(epochs, val_acc, "b-", label="loss")
plt.plot(epochs, val_acc_l2, "b--", label="the loss with l2")
plt.title("L2正则化对模型的影响")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

#dropout对过拟合的影响
