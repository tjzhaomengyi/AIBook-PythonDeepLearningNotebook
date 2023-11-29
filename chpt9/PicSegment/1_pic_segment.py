# -*- coding: utf-8 -*-
__author__ = 'Mike'
import os
import matplotlib.pyplot as plt
# from keras.utils import load_img, img_to_array
from PIL import Image
import numpy as np
import random
import keras
from keras import layers


input_dir = "/home/zhaomengyi/Projects/Datas/PicSegment/images"
target_dir = "/home/zhaomengyi/Projects/Datas/PicSegment/annotations/trimaps"
#线上
# input_dir = "/students/julyedu_693906/Projects/picsegment/images"
# target_dir = "/students/julyedu_693906/Projects/picsegment/annotations/trimaps"
#技巧：在数组中直接遍历生成，并排除不要的文件
input_img_paths = sorted([os.path.join(input_dir, fname)
                          for fname in os.listdir(input_dir)
                          if fname.endswith(".jpg")])
target_paths = sorted([
    os.path.join(target_dir, fname)
    for fname in os.listdir(target_dir)
    if fname.endswith(".png") and not fname.startswith(".")
])

img_size = (200, 200)
# target_size = (200, 200, 1)
num_imgs = len(input_img_paths)

#这个使用PIL.image.convert方法已经帮助我们转化成灰度格式的图片了，这里不需要再转换了，这个函数不用
def display_target(target_array):
    normalized_array = (target_array.astype("uint8") - 1) * 127
    plt.axis("off")
    plt.imshow(normalized_array[:, :, 0])


def path_to_input_image(path):
    return np.array(Image.open(path).resize(img_size))

def path_to_target(path):
    img = np.array(Image.open(path).resize(img_size).convert("L"))
    img = img.astype("uint8") - 1 #减1，使标签变成0、1、2
    return img

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))
    x = layers.Lambda(lambda x: x / 255)(inputs)
    #使用padding=same避免边界填充对特征图大小造成影响,
    #注意：只要大小变化就加上strides,变大的时候再大的上加stride，变小的时候也在大的上加
    '''下采样：3倍2次下采样，最终结果（25，25，256），使用strides代替之前cnn的maxpoolling操作'''
    x = layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(256, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    '''上采样：逆变换恢复原样'''
    x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same", strides=2)(x)
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    model = keras.Model(inputs, outputs)
    return model




plt.axis("off")
plt.imshow(Image.open(input_img_paths[9]))
plt.show()

img = Image.open(target_paths[9]).convert("L")
plt.imshow(img)
plt.show()
# display_target(img)

#将文件路径打乱
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_paths)
#将所有图像加载到float32格式的input_imgs数组中，将所有图像掩码加载到unit8格式的targets数组中（二者顺序相同）。输入有3个通道（RGB），目标只有一个通道（包含整数标签）
'''二元组相加结果，只能说python这个东西挺傻逼的，纯纯傻逼理解
(1337,) + (200, 200) + (1,)
Out[2]: (1337, 200, 200, 1)
'''
input_imgs = np.zeros((num_imgs,) + img_size + (3,), dtype="float32")
targets = np.zeros((num_imgs,) + img_size , dtype="uint8")
for i in range(num_imgs):
    try:
        input_imgs[i] = path_to_input_image(input_img_paths[i])
        targets[i] = path_to_target(target_paths[i])
    except:
        print(f"第{i}图片报错,{input_img_paths[i]},{target_paths[i]}")
        continue

num_val_samples = 1000
train_input_imgs = input_imgs[:-num_val_samples]
train_targets = targets[:-num_val_samples]
val_input_imgs = input_imgs[-num_val_samples:]
val_targets = targets[-num_val_samples:]


model = get_model(img_size=img_size, num_classes=3)
print(model.summary())
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
callbacks = [
    keras.callbacks.ModelCheckpoint("oxford_segment.keras", save_best_only=True)
]
#这里的targets类型的数据都是（200，200），要变成（200，200，1）才行
train_targets = np.stack([img.reshape(img.shape + (1,)) for img in train_targets])
val_targets = np.stack([img.reshape(img.shape + (1,)) for img in val_targets])

history = model.fit(train_input_imgs, train_targets, epochs=50, callbacks=callbacks, batch_size=64,
                    validation_data=(val_input_imgs, val_targets))

epochs = range(1, len(history.history["loss"]) + 1)
loss = history.history["loss"]
val_loss = history.history["val_loss"]
plt.figure()
plt.plot(epochs, loss, "bo", label="training loss")
plt.plot(epochs, val_loss, "b", label="validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()
plt.savefig("segment_loss.png")

