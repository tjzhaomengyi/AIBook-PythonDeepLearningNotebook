# -*- coding: utf-8 -*-
__author__ = 'Mike'
import keras
import numpy as np
from keras import layers
import matplotlib.pyplot as plt
import os, shutil, pathlib
from keras.preprocessing.image import ImageDataGenerator

'''
todo:迁移这个模型没有生成成功
'''

#提取VGG16的特征和对应标签，predict（）只接收图像作为输入，不接收labels，需要使用preprocessed_input进行预处理，这个函数将像素值缩放到合适的范围
#这里消耗时间非常长，preprocess_input()应该是这个函数
def get_features_and_labels(dataset):
    all_features = []
    all_labels = []
    for images, labels in dataset:
        preprocessed_images = keras.applications.vgg16.preprocess_input(images)
        features = conv_base.predict(preprocessed_images)
        all_features.append(features)
        all_labels.append(labels)
    return np.concatenate(all_features), np.concatenate(all_labels)

#weights指定模型初始化的权重检查点，include_top是否包含密集连接分类器，默认情况这个密集连接分类器对应ImageNet的1000个类别
conv_base = keras.applications.vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(180, 180, 3))
print(conv_base.summary)
original = "/home/zhaomengyi/Projects/Datas/Kaggle/DogVsCat/PetImages"
new_base = "/home/zhaomengyi/Projects/AIProjects/Book_PythonDeepLearn/chpt8/datas"
# new_base = "/students/julyedu_693906/Projects/datas" #线上地址
original_dir = pathlib.Path(original)
new_base_dir = pathlib.Path(new_base)
datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=0.1,
    zoom_range=0.2
) #也可以在原始这里面做scaling或者数据增强

train_dataset = datagen.flow_from_directory(new_base_dir / "train", target_size=(180, 180), batch_size=32, class_mode="binary")
validation_dataset = datagen.flow_from_directory(new_base_dir / "validation", target_size=(180, 180), batch_size=32, class_mode="binary")
test_dataset = datagen.flow_from_directory(new_base_dir / "test", target_size=(180, 180), batch_size=32, class_mode="binary")
train_features, train_labels = get_features_and_labels(train_dataset)
val_features, val_labels = get_features_and_labels(validation_dataset)
test_features, test_labels = get_features_and_labels(test_dataset)
print(train_features.shape)

#定义并训练密集连接分类器，我们自己的分类器
inputs = keras.Input(shape=(5, 5, 512)) #vgg16的输出是5，5，512
x = layers.Flatten()(inputs)
x = layers.Dense(26)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="relu")(x)
model = keras.Model(inputs, outputs)
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
callbacks = [keras.callbacks.ModelCheckpoint(
    filepath="feature_transfer.keras",
    save_best_only=True,
    monitor="val_loss"
)]
history = model.fit(
    train_features, train_labels,
    epochs=20,
    validation_data=(val_features, val_labels),
    callbacks=callbacks
)

#绘制结果
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.legend()
plt.savefig("transfer_acc.png")

plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Trainning and validation loss")
plt.legend()
plt.savefig("transfer_loss.png")
plt.show()