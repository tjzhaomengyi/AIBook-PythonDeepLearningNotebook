# -*- coding: utf-8 -*-
__author__ = 'Mike'
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#将图像调整为适当的数组
def preprocess_image(image_path):
    img = keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.vgg19.preprocess_input(img)
    return img

#将numpy数组转化为图像
def deprocess_image(img):
    img = img.reshape((img_height, img_width, 3))
    #vgg19.preprocess_input的作用是减去ImageNet平均像素值，使其中心为0.下面三行是一个逆操作
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1] #将图片的BGR格式转换为RGB格式，也是对vgg19.preprocess_input的逆操作
    img = np.clip(img, 0, 255).astype("uint8")
    return img

#内容损失函数，保证在vgg19网络靠近顶部的层来看，内容图像和组合图像很相似
def content_loss(base_img, combination_img):
    return tf.reduce_sum(tf.square(combination_img - base_img))

#定义风格损失函数
def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def style_loss(style_img, combination_img):
    S = gram_matrix(style_img)
    C = gram_matrix(combination_img)
    channels = 3
    size = img_height * img_width
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

#除了上面两个损失分量，还要添加第三个损失分量--总变差损失total variation loss,它对生成的组合图像的像素进行操作，使生成图像具有空间连续性
def total_variation_loss(x):
    a = tf.square(
        x[:,: img_height-1, : img_width-1, :] - x[:,1:, :img_width-1,:]
    )

    b = tf.square(
        x[:, : img_height - 1, : img_width - 1, :] - x[:, :img_height - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))


def compute_loss(combination_image, base_image, style_reference_image):
    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )
    features = feature_extractor(input_tensor)
    loss = tf.zeros(shape=())
    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(base_image_features, combination_features)

    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        style_loss_value = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layer_names)) * style_loss_value
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss

#梯度下降过程
@tf.function
def compute_loss_and_grads(combinantion_image, base_image, style_reference_image):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image, style_reference_image)
    grads = tape.gradient(loss, combination_image)
    return loss, grads

#梵高
base_image_path = keras.utils.get_file("sf.jpg", origin="https://img-datasets.s3.amazonaws.com/sf.jpg")
#城市图
style_reference_image_path = keras.utils.get_file("starry_night.jpg", origin="https://img-datasets.s3.amazonaws.com/starry_night.jpg")

original_width, original_height = keras.utils.load_img(base_image_path).size
#设置生成图片尺寸
img_height = 400
img_width = round(original_width * img_height / original_height)
'''
plt.imshow(keras.utils.load_img(base_image_path))
plt.show()
plt.figure()
plt.imshow(keras.utils.load_img(style_reference_image_path))
plt.show()
'''

#使用预训练的VGG19模型来创建一个特征提取器
model = keras.applications.vgg19.VGG19(weights="imagenet", include_top=False)#加载预训练的imagenet权重
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict) #特征提取器

#用风格损失的层列表
style_layer_names = [
    "block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"
]
content_layer_name = "block5_conv2" #用于内容损失的层
#总变差损失的贡献权重
total_variation_weight = 1e-6
style_weight = 1e-6 #风格损失的贡献权重
content_weight = 2.5e-8 #内容损失的贡献权重

optimizer = keras.optimizers.SGD(keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=100.0,
                                                                             decay_steps=100,
                                                                             decay_rate=0.96))
base_image = preprocess_image(base_image_path)
style_reference_image = preprocess_image(style_reference_image_path)
combination_image = tf.Variable(preprocess_image(base_image_path))

iterations = 4000
for i in range(1, iterations + 1):
    loss, grads = compute_loss_and_grads(combination_image, base_image, style_reference_image)
    optimizer.apply_gradients([(grads, combination_image)])
    if i % 100 == 0:
        print(f"Iteration {i}: loss={loss:.2f}")
        img = deprocess_image(combination_image.numpy())
        fname = f"combinantion_image_at_iteration_{i}.png"
        keras.utils.save_img(fname, img)


