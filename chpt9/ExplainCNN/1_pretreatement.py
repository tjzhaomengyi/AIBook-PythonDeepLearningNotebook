# -*- coding: utf-8 -*-
__author__ = 'Mike'
import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras import layers
import math


def get_img_array(img_path, target_size):
    img = Image.open(img_path).resize(target_size)
    array = np.array(img)
    #注意添加一个维度将三维图片变成一个一维的图片“批量”
    array = np.expand_dims(array, axis=0)
    return array

#1、查看只有一个图的图片批量
model = keras.models.load_model("../../chpt8/dog_vs_cat/convnet_from_scratch_with_augmentation.keras")
img_path = keras.utils.get_file(fname="cat.jpg", origin="https://img-datasets.s3.amazonaws.com/cat.jpg")
img_tensor = get_img_array(img_path, target_size=(180, 180))
plt.axis("off")
plt.imshow(img_tensor[0].astype("uint8"))
plt.show()

#2、实例化一个返回各层激活的模型
layer_outputs = []
layer_names = []
for layer in model.layers:
    if isinstance(layer, (layers.Conv2D, layers.MaxPooling2D)):
        layer_outputs.append(layer.output)
        layer_names.append(layer.name)
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs) #给定模型的输入，返回这个模型各层输出
#利用模型计算层的激活值
activations = activation_model.predict(img_tensor)
#给出输入图像的第一个卷积层的激活函数的结果
first_layer_activations = activations[0]
print(first_layer_activations.shape) #(1, 178, 178, 32),表示这是一张178 * 178、有32个通道的图
#将第五个通道可视化
plt.matshow(first_layer_activations[0, :, :, 5], cmap="viridis")
plt.show()

#将每个中间激活值的每个通道可视化
images_per_row = 16 #每16个像素算一次
for layer_name, layer_activation in zip(layer_names, activations):
    #每个激活函数的形状为（1， size， size， n_features）
    n_features = layer_activation.shape[-1] #通道数
    size = layer_activation.shape[1] #尺寸大小
    n_cols = n_features // images_per_row
    #准备一个空网格，来显示这个激活值中的所有通道
    display_grid = np.zeros((
        (size + 1) * n_cols - 1,
        images_per_row * (size + 1) - 1
    ))
    #在通道上遍历
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_index = col * images_per_row + row
            channel_image = layer_activation[0, :, :, channel_index].copy() #这是单个通道值或特征
            print(type(channel_image))
            if channel_image.sum() != 0:
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
            #将通道内图像的每个像素值压缩在0-255区间，如果小于0就是0，如果大于255就是255
            channel_image = np.clip(channel_image, 0, 255).astype("uint8")
            #将通道矩阵放入空网格中，
            display_grid[
                col * (size + 1) : (col + 1) * size + col,
                row * (size + 1) : (row + 1) * size + row
            ] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.axis("off")
    plt.imshow(display_grid, aspect="auto", cmap="viridis")
    plt.show()
'''
images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros(((size + 1) * n_cols - 1,
                             images_per_row * (size + 1) - 1))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_index = col * images_per_row + row
            channel_image = layer_activation[0, :, :, channel_index].copy()
            if channel_image.sum() != 0:
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype("uint8")
            display_grid[
                col * (size + 1): (col + 1) * size + col,
                row * (size + 1) : (row + 1) * size + row] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.axis("off")
    plt.imshow(display_grid, aspect="auto", cmap="viridis")
    plt.show()
'''