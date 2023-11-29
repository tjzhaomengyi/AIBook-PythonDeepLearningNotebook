# -*- coding: utf-8 -*-
__author__ = 'Mike'
import keras
from keras import layers
from PIL import Image
from keras.applications.xception import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.cm as cm
from scipy.ndimage import zoom

#注意：这个添加preprocess_input让模型的predict达到最准确
def get_img_array_for_xception(img_path, target_size):
    img = Image.open(img_path)
    img = img.resize(target_size)
    array = np.array(img)
    #注意添加一个维度将三维图片变成一个一维的图片“批量”
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array) #Xception的预处理，这样让加载的图像更好预测
    return array

#加载原始图，方便和热力图合并
def get_img_array(img_path, target_size):
    img = Image.open(img_path)
    img = img.resize(target_size)
    array = np.array(img)
    #注意添加一个维度将三维图片变成一个一维的图片“批量”
    array = np.expand_dims(array, axis=0)
    return array

#对img的数组格式进行放大操作
def get_img_arr_zoom(img, target_size):
    zoom_factor = (299 / 10, 299 / 10, 1)
    # zoom_img = np.resize(img, (target_size[0],target_size[1],target_size[2]))
    zoom_img = zoom(img, zoom_factor, order=1)
    return zoom_img



model = keras.applications.xception.Xception(weights="imagenet", include_top=True)
print(model.summary())
#输入图像是299*299
img_path = keras.utils.get_file(fname="elephant.jpg", origin="https://img-datasets.s3.amazonaws.com/elephant.jpg")
img_array = get_img_array_for_xception(img_path, target_size=(299, 299))
print(img_array.shape)
plt.axis("off")
plt.imshow(img_array[0])
plt.show()

preds = model.predict(img_array)
# [('n02504458', 'African_elephant', 0.8699273), ('n01871265', 'tusker', 0.076968335), ('n02504013', 'Indian_elephant', 0.02353712)]
print(keras.applications.xception.decode_predictions(preds, top=3)[0])
print(np.argmax(preds[0]))#类别号

#创建一个模型，返回最后一个卷积输出
last_conv_layer_name = "block14_sepconv2_act"
classifier_layer_names = ["avg_pool", "predictions"]
last_conv_layer = model.get_layer(last_conv_layer_name)
last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

#在最后一个卷积输出上再次应用分类器
last_conv_layer_output_shape = [dim.value for dim in last_conv_layer.output.shape[1:]] #注意：这里一定要转成arr，要不出现unhashable_dimisioin，从头就要开始转！！！！！操他妈的这里非常傻逼
print(f"平均层上一层的最后输出形状：{last_conv_layer_output_shape}")
classifier_input = keras.Input(shape=last_conv_layer_output_shape) #不包含tensor的第一个维度，表示数量,(10, 10, 2048)
x = classifier_input #把lasst_conv_layer的输出当做分类模型的输入
x = model.get_layer(classifier_layer_names[0])(x) #avg_pool层
print(f"avg_pooling结果{x.shape}")
pred_layer = model.get_layer(classifier_layer_names[1])#先获得输出层
print(pred_layer)
dimension_as_arr = [int(dim.value) for dim in x.shape[1:]]
x.set_shape((None,) + tuple(dimension_as_arr)) #这里要(None, 2048)
x = pred_layer(x)
classifier_model = keras.Model(inputs=classifier_input, outputs=x)

#检索做大预测类别的梯度
with tf.GradientTape() as tape:
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    last_conv_layer_output = last_conv_layer_model(img_tensor) # 到avg_pool的模型，倒数第三层的输出
    tape.watch(last_conv_layer_output)
    #检索与最大预测类别对应的激活通道
    preds = classifier_model(last_conv_layer_output) #最后两层，avg_pool + softmax分类器
    print(preds) #Tensor("model_2/predictions/Softmax:0", shape=(1, 1000), dtype=float32)
    top_pred_index = tf.argmax(preds[0])#是一个“一行1000列的结果”，选择出这个向量
    print(top_pred_index)
    top_class_channel = preds[:, top_pred_index]

grads = tf.gradients(top_class_channel, last_conv_layer_output) #这类最大预测类别相对于最后一个卷积层输出特征图的梯度

#梯度汇聚和通道重要性加权
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))#这是一个向量，其中每个元素使某个通道的平均梯度强度，它量化了每个通道对最大预测类别的重要性
# last_conv_layer_output = last_conv_layer_output
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 注意：这里一定要在这随机初始化以下
    pooled_grads = sess.run(pooled_grads)
    last_conv_layer_output = sess.run(last_conv_layer_output)
    last_conv_layer_output = last_conv_layer_output[0]
    print(pooled_grads.shape) #(10, 2048)
    print(last_conv_layer_output.shape)#(10, 2048)
#将最后一个卷积层输出的每个通道乘以pool_grads即该通道的重要性
for i in range(pooled_grads.shape[-1]):
    last_conv_layer_output[:, :, i] *= pooled_grads[:,i] #注意：last_conv_layer_output某个通道的值乘以该通道上的梯度权重，权重要选择（10，2048）列方向的值！！！
heatmap = np.mean(last_conv_layer_output, -1)#所得特征图的逐个通道均值就是类激活热力图

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
print(f"raw_heatmap:{heatmap.shape}")
plt.matshow(heatmap)
plt.show()

heatmap = np.uint8(255 * heatmap)
#使用jet对颜色进行重新着色
jet = cm.get_cmap("jet")
jet_colors = jet(np.arange(256))[:,:3]
jet_heatmap = jet_colors[heatmap]
plt.imshow(jet_heatmap)
plt.show()
print(type(jet_heatmap), jet_heatmap.shape)
print(type(img_array), img_array.shape)

#最后拿原生图
img_path = keras.utils.get_file(fname="elephant.jpg", origin="https://img-datasets.s3.amazonaws.com/elephant.jpg")
raw_img_array = get_img_array(img_path, target_size=(299, 299))
img = raw_img_array[0]
jet_heatmap = get_img_arr_zoom(jet_heatmap, target_size=(img.shape[1], img.shape[0], jet_heatmap.shape[2]))
plt.imshow(jet_heatmap)
plt.show()
print(f"jet_heatmap:{jet_heatmap.shape},{type(jet_heatmap)}. img_array:{img.shape},{type(img)}")


superimposed_img = jet_heatmap *20  + img#
print(superimposed_img)
plt.axis("off")
plt.imshow(superimposed_img.astype("uint8"))
plt.show()

image = Image.fromarray(superimposed_img.astype("uint8"))
image.save("result.png")
