# -*- coding: utf-8 -*-
__author__ = 'Mike'
from keras.datasets import imdb
import numpy as np
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt



#将sequeces进行0-1编码
def vectorize_sequences(sequences, dimension=10000):
    # 创建一个形状为len（(sequences),dimension）的矩阵
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results




# imdb的评论总共有88585个单词
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000) # 仅保留训练数据中签10000个最常出现的单词，低频次放弃
print(train_data[0]) # 这里面是评论词汇对应的词的编号
print(train_labels[0])
print(type(train_labels))

#将词典下载
word_index = imdb.get_word_index() #word_index是一个将单词映射为整数索引的字典
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])#将字典的键和值交换，将整数索引映射为单词
#索引减3，0 1 2分别是padding、start of sequence 和 unknown
decode_review = " ".join(reverse_word_index.get(i - 3, "?") for i in train_data[0])
print(decode_review)

#1、将训练数据向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
print(x_train[0])
y_train = np.asarray(train_labels).astype("float32")#这里就是脱裤子放屁直接astype即可
y_test = np.asarray(test_labels).astype("float32")
print(type(y_train))

#2、构建模型
# 为什么要加入激活函数？仿射变换只能学到数据的线性变换，假设空间非常受限，无法利用多个表示层的优势，增加层数并不能扩展假设空间。
#  为了得到更多更丰富的假设空间，从而充分利用多层表示的优势，需要引入非线性，也就是添加激活函数。
model = keras.Sequential([
    layers.Dense(32, activation="relu"),#把输入的10000维度数据映射到16维度的空间上
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

#3、验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#4、训练模型
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

#5、看看训练结果
print(type(history))
history_dict = history.history
print(history_dict.keys())

#训练损失和验证损失
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"] #验证损失
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#训练精度和验证精度
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="training acc")
plt.plot(epochs, val_acc, "b", label="validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

#测试集
results = model.evaluate(x_test, y_test)
print(results) #输出测试损失和测试精度
