# -*- coding: utf-8 -*-
__author__ = 'Mike'
import keras
import numpy as np
from keras import layers
import matplotlib.pyplot as plt
import os, shutil, pathlib
from keras.preprocessing.image import ImageDataGenerator

#这个不用那个vgg的向前传播，finetune的效果挺好

# new_base = "/home/zhaomengyi/Projects/AIProjects/Book_PythonDeepLearn/chpt8/datas"
new_base = "/students/julyedu_693906/Projects/datas" #线上地址

new_base_dir = pathlib.Path(new_base)
datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=0.1,
    zoom_range=0.2
) #也可以在原始这里面做scaling或者数据增强

train_dataset = datagen.flow_from_directory(new_base_dir / "train", target_size=(180, 180), batch_size=32, class_mode="binary")
validation_dataset = datagen.flow_from_directory(new_base_dir / "validation", target_size=(180, 180), batch_size=32, class_mode="binary")
test_dataset = datagen.flow_from_directory(new_base_dir / "test", target_size=(180, 180), batch_size=32, class_mode="binary")


conv_base = keras.applications.vgg16.VGG16(
    weights="imagenet", include_top=False
)
conv_base.trainable = True
#把倒数第四层之前的模型都冻住
for layer in conv_base.layers[:-4]:
    layer.trainable = False

inputs = keras.Input(shape=(180, 180, 3))
x = keras.layers.Lambda(lambda x : x / 255.0)(inputs)
x = conv_base(x)
x = layers.Flatten()(x)
x = layers.Dense(256)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(loss="binary_crossentropy",
              optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
              metrics=["accuracy"])
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="feature_extraction_with_data_augmentation.keras",
        save_best_only=True,
        monitor="val_loss"
    )
]
history = model.fit(train_dataset, epochs=30, validation_data=validation_dataset, callbacks=callbacks)


acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.legend()
plt.savefig("finetune_acc.png")

plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Trainning and validation loss")
plt.legend()
plt.savefig("finetune_loss.png")
plt.show()