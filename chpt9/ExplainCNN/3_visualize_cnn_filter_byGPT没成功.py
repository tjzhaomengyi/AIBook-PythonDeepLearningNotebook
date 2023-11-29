import tensorflow as tf
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt

# 加载预训练的 Xception 模型
base_model = Xception(weights='imagenet')

# 选择模型的某一层进行可视化
layer_name = 'block4_sepconv2_act'
layer = base_model.get_layer(layer_name)

# 构建一个新模型，该模型输出指定层的激活值
model = tf.keras.Model(inputs=base_model.input, outputs=layer.output)

# 定义需要最大化的损失函数，即选定层的激活值
def compute_loss(input_image, filter_index):
    activation = model(input_image)
    return tf.reduce_mean(activation[:, :, :, filter_index])

# 定义梯度上升函数
def gradient_ascent_step(img, filter_index, learning_rate=0.01):
    loss = compute_loss(img, filter_index)
    grads = tf.gradients(loss, img)
    grads /= tf.maximum(tf.reduce_mean(tf.abs(grads)), 1e-6)
    img += learning_rate * grads[0]
    return img

# 定义滤波器可视化函数
def visualize_filter(layer_name, filter_index, iterations=100, learning_rate=0.01):
    img_height = 224
    img_width = 224

    # 生成随机输入图像
    img = np.random.random((1, img_height, img_width, 3)) * 20 + 128.

    # 将输入图像进行预处理，使其符合 Xception 模型的输入要求
    img = preprocess_input(img)

    # 转换为 TensorFlow 张量
    img = tf.convert_to_tensor(img, dtype=tf.float32)

    # 获取选定层的激活值
    layer = model.get_layer(layer_name)
    layer_activation = layer.output

    # 最大化选定层的激活值
    for iteration in range(iterations):
        img = gradient_ascent_step(img, filter_index, learning_rate)

    # 后处理，将图像转换为有效的图像格式
    with tf.Session() as sess:
        img = sess.run(img).squeeze()
        img -= img.min()
        img /= img.max()
        img *= 255

    return img

# 选择要可视化的滤波器索引
filter_index = 0  # 可以根据需要更改

# 可视化滤波器
visualization = visualize_filter(layer_name, filter_index)

# 显示可视化结果
plt.imshow(visualization.astype('uint8'))
plt.show()
plt.savefig("visualize_filter.png")