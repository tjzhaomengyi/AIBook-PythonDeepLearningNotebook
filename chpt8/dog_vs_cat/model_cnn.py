# -*- coding: utf-8 -*-
__author__ = 'Mike'
from keras import layers
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os, shutil, pathlib
import matplotlib.pyplot as plt
import keras

#增加数据增强，新版本可以用layers.RandomFilp方式进行增项
# data_augmentation = keras.Sequential([
#
# ])
inputs = keras.Input(shape=(180, 180, 3))
x = keras.layers.Lambda(lambda x : x / 255.0)(inputs) #新版本的keras用layers.Rescaling替代
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x) #在数据预测的时候dropout不起作用
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

original = "/home/zhaomengyi/Projects/Datas/Kaggle/DogVsCat/PetImages"
# new_base = "/home/zhaomengyi/Projects/AIProjects/Book_PythonDeepLearn/chpt8/datas"
new_base = "/students/julyedu_693906/Projects/datas" #线上地址
original_dir = pathlib.Path(original)
new_base_dir = pathlib.Path(new_base)
# tensorflow2.24版本以上有这个函数
# train_data = image_dataset_from_directory(new_base_dir / "train", image_size(180, 180), batch_size=32)
# tensorflow1.14
datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=0.1,
    zoom_range=0.2
) #也可以在原始这里面做scaling或者数据增强

train_dataset = datagen.flow_from_directory(new_base_dir / "train", target_size=(180, 180), batch_size=32, class_mode="binary")
validation_dataset = datagen.flow_from_directory(new_base_dir / "validation", target_size=(180, 180), batch_size=32, class_mode="binary")
test_dataset = datagen.flow_from_directory(new_base_dir / "test", target_size=(180, 180), batch_size=32, class_mode="binary")
for data_batch, labels_batch in train_dataset:
    print("data batch shape:", data_batch.shape)
    print("labels batch shape:", labels_batch.shape)
    print(labels_batch)
    break

#callbacks负责监控迭代val_loss，如果这个有变化才会更新保存的模型
callbacks = [keras.callbacks.ModelCheckpoint(filepath="convnet_from_scratch_with_augmentation.keras", save_best_only=True, monitor="val_loss")]

history = model.fit(train_dataset, epochs=100, validation_data=validation_dataset, callbacks=callbacks)
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.savefig("acc_augment.png")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and Validation loss")
plt.legend()
plt.savefig("loss_augment.png")
plt.show()

