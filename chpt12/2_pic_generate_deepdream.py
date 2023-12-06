# -*- coding: utf-8 -*-
__author__ = 'Mike'
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras.applications import inception_v3
import tensorflow as tf
import numpy as np



#接下来计算损失，即在每个处理尺度的梯度上升过程中需要最大化的量。【回忆：在chpt9中的滤波器可视化实例中，我们试图将某一层某个滤波器的值最大化。】
# 这里我们要将多个层的全波滤波器激活值同时最大化。具体来说，就是对一组靠近顶部的层的激活值的L2范数进行加权求和，然后将其最大化。
# 选择哪些层以及它们对最终损失的贡献大小，对生成的可视化效果有很大影响，所以我们希望让这些参数易于配置。更靠近底部的层级生成的是几何图案
# 而更靠近顶部的层生成的则是从中能看到某些ImageNet类别（比如鸟和狗）的图案
def compute_loss(input_image):
    features = feature_extractor(input_image)
    loss = tf.zeros(shape=())
    for name in features.keys():
        coeff = layer_settings[name]
        activation = features[name]
        loss += coeff * tf.reduce_mean(tf.square(activation[:, 2:-2, 2:-2, :])) #为了避免出现边界伪影，损失中只包含非边界像素
    return loss

@tf.function
def gradient_ascent_step(image, learning_rate):
    with tf.GradientTape() as tape: #计算DeepDream损失相对于当前图像的梯度
        tape.watch(image)
        loss = compute_loss(image)
    grads = tape.gradient(loss, image)
    grads = tf.math.l2_normalize(grads)
    image += learning_rate * grads
    return loss, image

#在图像尺寸（八度）上运行梯度上升，max_loss是损失最大
def gradient_ascent_loop(image, iterations, learning_rate, max_loss=None):
    for i in range(iterations):
        loss, image = gradient_ascent_step(image, learning_rate)
        if max_loss is not None and loss > max_loss: #提前终止
            break
        print(f"...Loss value at step{i}: {loss:.2f}")
    return image


def proprocess_image(image_path):
    img = keras.utils.load_img(image_path)
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.inception_v3.preprocess_input(img)
    return img

#这个可以将numpy数组转换为有效图像
def deprocess_image(img):
    img = img.reshape((img.shape[1], img.shape[2], 3))
    img /= 2.0
    img += 0.5
    img *= 255.
    img = np.clip(img, 0, 255).astype("uint8")
    return img



base_image_path = keras.utils.get_file("coast.jpg", origin="https://img-datasets.s3.amazonaws.com/coast.jpg")

plt.axis("off")
plt.imshow(keras.utils.load_img(base_image_path))
plt.show()

model = inception_v3.InceptionV3(weights="imagenet", include_top=False)

#我们试图将这些层的激活函数值最大化。这里给出了这些层在总损失中所占的权重，你可以通过调整这些值来得到新的视觉效果
layer_settings = {
    "mixed4": 2.0,
    "mixed5": 3.5,
    "mixed6": 0.5,
    "mixed7": 0.2
}
outputs_dict = dict([
    (layer.name, layer.output)
    for layer in [model.get_layer(name) for name in layer_settings.keys()]
])
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

step = 20. #梯度上升的步长
num_octave = 3 #在几个八度上进行梯度上升
octave_scale = 1.4 # 连续八度之间的尺寸比例
iterations = 30 #没饿个尺寸上运行梯度上升的步数
max_loss =14.

original_img = proprocess_image(base_image_path) #加载图像
original_shape = original_img.shape[1:3]

successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes = successive_shapes[::-1]

shrunk_original_img = tf.image.resize(original_img, successive_shapes[0])

img = tf.identity(original_img) #复制图像
for i, shape in enumerate(successive_shapes):
    print(f"proccessing octave{i} with shape{shape}")
    # proccessing octave0 with shape(459, 612) proccessing octave1 with shape(642, 857)
    img = tf.image.resize(img, shape) #此时Image的shape(1,459, 612,3)
    img = gradient_ascent_loop(img, iterations=iterations, learning_rate=step, max_loss=max_loss)
    upscaled_shrunk_original_img = tf.image.resize(shrunk_original_img, shape)
    same_size_original = tf.image.resize(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img
    img += lost_detail
    shrunk_original_img = tf.image.resize(original_img, shape)

keras.utils.save_img("dream_1.png", deprocess_image(img.numpy())) #这个时候img.numpy()的形状在结束的时候为（1，x,y,3）


