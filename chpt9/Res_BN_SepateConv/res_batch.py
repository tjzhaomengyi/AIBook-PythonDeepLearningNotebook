# -*- coding: utf-8 -*-
__author__ = 'Mike'
import keras
from keras import layers
'''
情况1：滤波器数量发生变化时的残差块，x没有做最大池化
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, activation="relu")(inputs)
residual = x
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
residual = layers.Conv2D(64, 1)(residual)#todo:这里比较奇怪，直接从32*32大小投影成64*64的残差，但是下面的方法使用strides=2进行恢复
x = layers.add([x, residual]) #输出块和residual都是（64 * 64），可以相加

情况2：目标层块包含最大汇池化的情况
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, activation="relu")(inputs)
residual = x
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(2, padding="same")(x)
residual = layers.Conv2D(64, 1, strides=2)(residual)
x = layers.add([x, residual])
'''

inputs = keras.Input(shape=(32, 32, 3))
x = layers.Lambda(lambda x : x / 255)(inputs)

#注意：一个实用的残差函数，用于实现带有残差连接的卷积损失层块，可选择添加最大汇聚
def residual_block(x, filters, pooling=False):
    residual = x
    x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
    if pooling:
        x = layers.MaxPooling2D(2, padding="same")(x)
        residual = layers.Conv2D(filters, 1, strides=2)(residual)
    elif filters != residual.shape[-1]:
        residual = layers.Conv2D(filters, 1)(residual) #直接进行投影
    x = layers.add([x, residual])
    return x

x = residual_block(x, filters=32, pooling=True)
x = residual_block(x, filters=64, pooling=True)
x = residual_block(x, filters=128, pooling=False) #最后一层不需要最大池化层，因为后面马上要将其进行全局平均汇聚

x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())

#格式化的批量归一化
'''
注意由于规范化会将该层输出的均值设为0，因此在使用BN时候不再需要偏置向量，使用use_bias=False不加偏置，
然后在BN层在加激活层，并加入偏置项.这样做，最大限度利用relu。如果先做卷积，再做激活然后批量归一化也行    
'''
x = layers.Conv2D(32, 3, use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("Relu")(x)