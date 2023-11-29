# -*- coding: utf-8 -*-
__author__ = 'Mike'
import keras
from keras import layers
import matplotlib.pyplot as plt
import os, shutil, pathlib
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# new_base = "/home/zhaomengyi/Projects/AIProjects/Book_PythonDeepLearn/chpt8/datas"
new_base = "/students/julyedu_693906/Projects/Datas/dog_and_cat" #线上地址

new_base_dir = pathlib.Path(new_base)
datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=0.1,
    zoom_range=0.2
) #也可以在原始这里面做scaling或者数据增强

train_dataset = datagen.flow_from_directory(new_base_dir / "train", target_size=(180, 180), batch_size=32, class_mode="binary")
validation_dataset = datagen.flow_from_directory(new_base_dir / "validation", target_size=(180, 180), batch_size=32, class_mode="binary")
test_dataset = datagen.flow_from_directory(new_base_dir / "test", target_size=(180, 180), batch_size=32, class_mode="binary")

inputs = keras.Input(shape=(180, 180, 3))
x = layers.Lambda(lambda x : x / 255)(inputs)
x = layers.Conv2D(filters=32, kernel_size=5, use_bias=False)(x)

'''
注意：可卷积分离的底层假设，即特征通道在很大程度上是不相关的，对于RGB图像来说并不成立。
    在自然图像中，红绿蓝这三个颜色通道实际上是高度相关的，因此模型的第一层是常规的cov2d，之后使用分离卷积
'''
for size in [32, 64, 128, 256, 512]:
    residual = x
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x) #卷积分离

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    residual = layers.Conv2D(size, 1, strides=2, padding="same", use_bias=False)(residual)
    x = layers.add([x, residual])

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(loss="binary_crossentropy",
              optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
              metrics=["accuracy"])
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="small_xception.keras",
        save_best_only=True,
        monitor="val_loss"
    )
]
history = model.fit(train_dataset, epochs=100, validation_data=validation_dataset, callbacks=callbacks)


acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.legend()
plt.savefig("small_xception_acc.png")

plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Trainning and validation loss")
plt.legend()
plt.savefig("small_xception_loss.png")
plt.show()