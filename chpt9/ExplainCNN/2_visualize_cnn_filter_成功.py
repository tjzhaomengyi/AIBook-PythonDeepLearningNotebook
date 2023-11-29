# -*- coding: utf-8 -*-
__author__ = 'Mike'
import keras
from keras import layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.applications.xception import preprocess_input

# tf.enable_eager_execution()
'''
model.predict() 和 Model(x)的区别，两个都表示在x上运行模型并检索输出y。predict可以对数据进行批量循环，可以扩展到非常大的数组。
model(x) 发生在内存中，无法扩展。此外，predict是不可微的，就是一个简单的前向传播。如果要检索梯度就应该使用model(x).
总之，除非写底层循环，否则前向传播直接用predict（）即可。
'''
def compute_loss(image, filter_index):
    # image = keras.applications.xception.preprocess_input(image)
    activation = feature_extractor(image)
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]#todo:
    return tf.reduce_mean(filter_activation) #返回激活函数的均值

#通过梯度上升实现损失最大化
# @tf.function
def gradient_ascent_step(image, filter_index, learning_rate=0.01):
    with tf.GradientTape() as tape:
        tape.watch(image) #明确监控图像质量，因为它不是一个TensorFlow Variabe，注意：在梯度带中只会自动监控Variable
        loss = compute_loss(image, filter_index) #计算损失标量，表示当前图像对滤波器的激活程度
    grads = tf.gradients(loss, image) #计算损失相对于图像的梯度
    #grads = tape.gradient(loss, image)
    grads = tf.math.l2_normalize(grads) #应用梯度规范化技巧
    # grads /= tf.maximum(tf.reduce_mean(tf.abs(grads)), 1e-6)
    image += learning_rate * grads[0] #将图像沿着能够更强烈激活目标滤波器的方向移动一小步
    return image

#生成可视化滤波器的函数
def generate_filter_pattern(filter_index):
    img_width = 200
    img_height = 200
    iterations = 30 #梯度上升步数
    learning_rate = 10.
    image = tf.random.uniform(
        minval=0.4,
        maxval=0.6,
        shape=(1, img_width, img_height, 3))
    #todo:
    image = preprocess_input(image)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    for i in range(iterations):
        image = gradient_ascent_step(image, filter_index, learning_rate)
    print(type(image))
    # with tf.Session() as sess:
    #     image = sess.run(image)
    return image

#将图片数组中的值压缩在[0, 255]
def deprocess_image(image):
    image -= image.mean()
    image /= image.std()
    image *= 64
    image += 128
    image = np.clip(image, 0, 255).astype("uint8")
    image = image[25:-25, 25:-25, :]
    return image

def get_img_array(img_path, target_size):
    img = Image.open(img_path).resize(target_size)
    array = np.array(img)
    #注意添加一个维度将三维图片变成一个一维的图片“批量”
    array = np.expand_dims(array, axis=0)
    return array

model = keras.applications.xception.Xception(weights="imagenet", include_top=False)
#打印xception模型cnn和卷积分离层的名称
for layer in model.layers:
    if isinstance(layer, (keras.layers.Conv2D, keras.layers.SeparableConv2D)):
        print(layer.name)

#创建特征提取器模型，返回某一层的输出
layer_name = "block3_sepconv1" #这个是xception模型中的任意一层block3_sepconv1
layer = model.get_layer(name=layer_name) #构建层对象
feature_extractor = keras.Model(inputs=model.input, outputs=layer.output) #直接使用该层生成模型对象，用model输入和输出

# img_path = keras.utils.get_file(fname="cat.jpg", origin="https://img-datasets.s3.amazonaws.com/cat.jpg")
# img_tensor = get_img_array(img_path, target_size=(180, 180))
#在使用这个模型的时候需要使用keras.applications.xception.preprocess_input函数进行预处理
# img_tensor = keras.applications.xception.preprocess_input(img_tensor)
# activation = feature_extractor.predict(img_tensor)

plt.axis("off")

#随机生成一个图片，然后进行可视化滤波器
gen_filter_img = generate_filter_pattern(filter_index=2)
print(gen_filter_img)
with tf.Session() as sess:
    # 使用 run 方法运行 Session 获取 NumPy 数组，返回的是一个类似模型的文件，所有这里要初始化全局参数，然后再生成视觉化的滤波器
    sess.run(tf.global_variables_initializer()) #注意：这里一定要在这随机初始化以下
    gen_filter_img = sess.run(gen_filter_img)
print(gen_filter_img[0])
plt.imshow(deprocess_image(gen_filter_img[0]))
plt.show()
