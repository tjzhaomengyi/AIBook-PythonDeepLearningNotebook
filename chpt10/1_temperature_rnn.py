# -*- coding: utf-8 -*-
__author__ = 'Mike'
import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers



fname = os.path.join("/home/zhaomengyi/Projects/Datas/RNN_Datas/jena_climate_2009_2016.csv")

with open(fname) as f:
    data = f.read()
lines = data.split("\n")
#['"Date Time"', '"p (mbar)"', '"T (degC)"', '"Tpot (K)"', '"Tdew (degC)"', '"rh (%)"', '"VPmax (mbar)"',
#  '"VPact (mbar)"', '"VPdef (mbar)"', '"sh (g/kg)"', '"H2OC (mmol/mol)"', '"rho (g/m**3)"',
# '"wv (m/s)"', '"max. wv (m/s)"', '"wd (deg)"']
header = lines[0].split(",")
lines = lines[1:]
print(header)
print(len(lines))

#探索数据
temperature = np.zeros((len(lines), ))#单独一个温度向量
raw_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(",")[1:]]
    temperature[i] = values[1] #将温度保存在温度数组中
    raw_data[i, :] = values[:] #将剔除时间数据的其余数据放入数组

plt.plot(range(len(temperature)), temperature)
#绘制前十天数据曲线，数据每10分钟记录一次，十天数据有1440=24*6*10个
plt.figure()
plt.plot(range(1440), temperature[:1440])
# plt.show()

'''
我们以天为尺度，看看这个时间序列是否可以预测。使用50%数据用于训练，25%数据用于验证，25%数据用于测试
注意：验证数据和测试数据一定要比训练数据集合更靠后
'''
num_train_samples = int(0.5 * len(raw_data))
num_val_samples = int(0.25 * len(raw_data))
num_test_samples = len(raw_data) - num_val_samples - num_train_samples
print(f"num_train_samples:{num_train_samples}")
print(f"num_val_samples:{num_val_samples}")
print(f"num_test_samples:{num_test_samples}")

'''
具体问题描述：每小时采样一次数据，给定前五天数据，能否预测24小时之后的温度
'''
#技巧1：在时间序列数据上做规范化
mean = raw_data[:num_train_samples].mean(axis=0) #针对每列进行规范化汇总
raw_data -= mean
std = raw_data[:num_train_samples].std(axis=0)
raw_data /= std
#创建一个Dataset对象，可以生成过去5天的数据批量，以及24小时之后的目标温度。
#样本N和样本N+1，二者数据大部分时间是相同的，我们实时生成样本，保存最初的raw_data和temperature
# 使用keras内置的timeseries_dataset_from_array()完成这个工作，本质就是对一个数组滑窗取值，其中参数
# sequence_length表示滑窗长度，target数组表示目标滑动数组的起始位置的数组
'''
import numpy as np
from tensorflow import keras

int_sequence = np.arange(10)
dummy_dataset = 8.0(
    data=int_sequence[:-3],
    targets=int_sequence[3:],
    import numpy as np
from tensorflow import keras

int_sequence = np.arange(10) #[0,1,2,3,4,5,6,7,8,9]
dummy_dataset = keras.utils.timeseries_dataset_from_array(
    data=int_sequence[:-3],    #从[0,1,2,3,4,5,6]中生成结果
    targets=int_sequence[3:],  #从位置3开始生成结果
    sequence_length=3,
    batch_size=2,
)
print(dummy_dataset)
for inputs, targets in dummy_dataset:
    for i in range(inputs.shape[0]):
        print([int(x) for x in inputs[i]], int(targets[i]))
target表示输出位置的结果，表示从[0,1,2]后可以得到3
[0, 1, 2] 3 
[1, 2, 3] 4
[2, 3, 4] 5
[3, 4, 5] 6
[4, 5 ,6] 7 
'''

sampling_rate = 6
sequence_length = 120
delay = sampling_rate * (sequence_length + 24 -1)
batch_size = 256


train_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True, #让256个批次的数值乱序
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples)

val_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples
)
test_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples + num_val_samples
)

#每个数据集都会生成(samples,target）,其中samples是包含256个样本的批量，每个样本包含连续120小时的输入数据；targets是包含相应的256个
#目标温度的数组。因为样本被随机打乱，所以samples[n]和samples[n+1]不一定在时间上最接近
# 技巧：train_dataset是一个BatchDataset的数据类型可以是使用map(lambda x,y : (x,y))获取样本和目标，也可以是用unbatch拆分元祖
for samples, targets in train_dataset:#迭代每一批次的样本
    print("sample shape", samples.shape) #(256,120,14) 理解：256个批量样本中，每个样本中有120个时间序列样本，每个单位时间的样本有14个维度特征
    print("target shape", targets.shape) #(256,)
    break

'''
方法1：基于MAE的机器学习预测。
假设：温度时间序列是连续的，明天的温度很可能接近今天的温度
def evaluate_naive_method(dataset):
    total_abs_err = 0
    samples_seen = 0
    for samples, targets in dataset:
        #温度特征在第一列，所以samples[:, -1, 1]是输入序列最后一个温度的测量值，之前我们对特征进行了规范化，所以要得到以摄氏度为单位的温度值
        #还需要乘以标准化并加上均值
        preds = samples[:, -1, 1] * std[1] + mean[1]
        total_abs_err += np.sum(np.abs(preds - targets)) #计算所有插值
        samples_seen += samples.shape[0]#统计数量
    return total_abs_err / samples_seen#MAE

print(f"Validation MAE:{evaluate_naive_method(val_dataset):.2f}") #2.44
print(f"Test MAE:{evaluate_naive_method(test_dataset):.2f}")#2.62
'''

'''
方法2：使用小型机器学习
（1）将数据展平 （2）使用两个Dense层。使用均方误差作为损失，不使用MAE

inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1])) #(120,14)
x = layers.Flatten()(inputs)
x = layers.Dense(16, activation="relu")(x)
outputs = layers.Dense(1)(x) #回归问题，不需要激活函数
model = keras.Model(inputs, outputs)

callbacks = [keras.callbacks.ModelCheckpoint("jena_dense.keras", save_best_only=True)]
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=callbacks)
model = keras.models.load_model("jena_dense.keras")
print(f"Test MAE:{model.evaluate(test_dataset)[1]:.2f}")

loss = history.history["mae"]
val_loss = history.history["val_mae"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training MAE")
plt.plot(epochs, val_loss, "b", label="Val MAE")
plt.title("Training and Val MAE")
plt.legend()
plt.show()
'''
'''
方法3：使用一维卷积神经网络。根据窗口平滑不变性假设的序列数据。在一维数据上，如果沿着序列滑动一个窗口，
那么窗口的内容应该遵循相同的属性，与窗口位置无关.
这个假设根据直觉明显就是错误的，温度的时间序列明显和时间相关，每个窗口剔除出来的数据不可能是不变的，
并且汇聚层会忽略掉一些信息，但是这些信息都是预测是需要的关键信息
'''
'''
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.Conv1D(8, 24, activation="relu")(inputs) #选24小时的数据集
x = layers.MaxPooling1D(2)(x)
x = layers.Conv1D(8, 12, activation="relu")(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Conv1D(8, 6, activation="relu")(x)
x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

callbacks = [keras.callbacks.ModelCheckpoint("jena_conv.keras", save_best_only=True)]
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset,
                    callbacks=callbacks)
model = keras.models.load_model("jena_conv.keras")
print(f"Test MAE:{model.evaluate(test_dataset)[1]:.2f}")

loss = history.history["mae"]
val_loss = history.history["val_mae"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training MAE")
plt.plot(epochs, val_loss, "b", label="Val MAE")
plt.title("Training and Val MAE")
plt.legend()
plt.show()
'''
'''
方法4：基于LSTM的RNN模型
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.LSTM(16)(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
callbacks = [keras.callbacks.ModelCheckpoint("jena_lstm.keras", save_best_only=True)]
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=callbacks)
model = keras.models.load_model("jena_lstm.keras")
print(f"Test MAE:{model.evaluate(test_dataset)[1]:.2f}")

loss = history.history["mae"]
val_loss = history.history["val_mae"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training MAE")
plt.plot(epochs, val_loss, "b", label="Val MAE")
plt.title("Training and Val MAE")
plt.legend()
plt.show()
'''

'''
使用Numpy实现RNN
timesteps = 100 #输入序列的时间步数
input_features = 32  #输入特征空间的维度
output_features = 64 #输出特征空间的维度
inputs = np.random.random((timesteps, input_features))
state_t = np.zeros((output_features,))
W = np.random.random((output_features, input_features)) #输入权重矩阵
#注意上一层的ouput作为在这一层额state
U = np.random.random((output_features, output_features))#输入状态权重矩阵
b = np.random.random((output_features,))
successive_outputs = []
for input_t in inputs:
    #遍历输入的每个向量（64，）
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t
final_output_sequence = np.stack(successive_outputs, axis=0) #最终输出一个形状为(timesteps,output_features)的2阶张量
print(final_output_sequence)
'''
'''
可以处理任意长度的RNN层,这样模型可以处理任意长度的序列。但是，如果所有序列的长度相同，应该指定出完整
的输入形状，这样可以完整表达模型输出长度信息，这样可以解锁RNN的性能优化。
inputs = keras.Input(shape=(None, num_features))
Keras 中的左右循环层(SimpleRNN层、LSTM层和GRU层)都可以在两种模式下运行：
    一种是返回每个时间步连续输出的完整序列形状为（batch_size,timesteps, output_features）三阶向量
    另一种是返回每个输入序列的最终输出，即形状为(batch_size,output_features)的二阶向量
'''
# num_features = 14
# steps = 120
# inputs = keras.Input(shape=(steps, num_features))
# outputs = layers.SimpleRNN(16,return_sequences=False)(inputs)
# print(outputs.shape) #(None, 16)
'''
SimpleRNN的问题是长期依赖存在梯度小时问题。
LSTM在这个问题上得到解决，处理原理：保存信息便后续使用，防止较早的信号在处理过程中逐渐消失，类似残差连接
1、LSTM的第一步增加携带轨道，添加跨时间步的信息c_t
c_t的变换和SimpleRNN的变化一致:y = activate(dot(state_t, U) + dot(input_t, W) + b)
但是是哪个变化有自己的权重矩阵，分别用i、f、k作为下表
'''
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
#技巧在使用cpu训练模型的时候，可以把这里进行展开unroll=true
x = layers.LSTM(32, recurrent_dropout=0.25)(inputs)
x = layers.Dropout(0.5)(x) #进行正则化
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
callbacks = [keras.callbacks.ModelCheckpoint("jena_lstm_dropout.keras", save_best_only=True)]
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset, epochs=50, validation_data=val_dataset, callbacks=callbacks)

loss = history.history["mae"]
val_loss = history.history["val_mae"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training MAE")
plt.plot(epochs, val_loss, "b", label="Val MAE")
plt.title("Training and Val MAE")
plt.legend()
plt.show()
plt.savefig("LSTM_MAE.png")

