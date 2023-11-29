# -*- coding: utf-8 -*-
__author__ = 'Mike'
from keras.datasets import reuters
from keras.utils import to_categorical
import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

#将sequeces进行0-1编码
def vectorize_sequences(sequences, dimension=10000):
    # 创建一个形状为len（(sequences),dimension）的矩阵
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results

#新闻是多分类，对label进行0-1编码
#这个函数等价于keras的方法：keras.utils.to_categorical
def to_one_hot(labels, dimension=46):
    results = np.zeros(len(labels), dimension)
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
print(len(train_data))
print(len(test_data))

#看一看第一个新闻的内容
word_index = reuters.get_word_index() #(word, index)
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire =  " ".join([reverse_word_index.get(i - 3, "?") for i in train_data[0]])
print(decoded_newswire)

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

#构建模型，16为空间过小，我们要区分46个类别，这里使用64个单元,[1.17681553697119, 0.7943009734153748]
#
model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    # layers.Dense(64, activation="relu"),
    layers.Dense(46, activation="softmax")
])
#模型的反向传播
model.compile(optimizer="rmsprop", loss="categorical_crossentropy",metrics=["accuracy"])
#验证集
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]
#训练模型
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
#模型效果
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.clf()#清空图像
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.plot(epochs, acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

result = model.evaluate(x_test, y_test) #返回测试集的损失和准确率
print(result)

predictions = model.predict(x_test)
print(np.argmax(predictions[0]), test_labels[0])
