# -*- coding: utf-8 -*-
__author__ = 'Mike'
from keras.datasets import boston_housing
from keras import layers
import keras
import numpy as np
import matplotlib.pyplot as plt

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])
    #MSE-均方差，MAE-平均绝对误差
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model



(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print(train_data.shape)
print(test_data.shape)
print("和最后的结果进行比较 " + str(test_targets[0]))

#标准化,注意：对测试数据进行标准化的指标使用的是训练数据集的标准化指标，不能使用测试集自己的指标！！！
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

#这个例子的样本数量非常少，使用K折交叉验证
k = 4 #将数据划分为4个部分，一个部分用于验证另外三个部分用于训练，然后这四个部分分别作为验证，进行四次训练
num_val_samples = 100
num_epochs = 500
all_scores = []
all_mae_histories = []
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    print(len(val_data))
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

    model = build_model()
    #verbose=0静默模式,就是不打印连着的训练过程的进度条
    history = model.fit(partial_train_data, partial_train_targets,
                        epochs=num_epochs, batch_size=16, verbose=0,validation_data=(val_data, val_targets))
    mae_history = history.history["val_mae"]
    all_mae_histories.append(mae_history)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
print(all_scores) #[2.099381446838379, 2.5528602600097656, 2.4368505477905273, 2.455676794052124]
#计算每轮所有MAE的平均值
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel("Epochs")
plt.ylabel("validation MAE")
plt.show()

#剔除前面的10个数据点
truncated_mae_history = average_mae_history[10:]
plt.plot(range(1, len(truncated_mae_history) + 1), truncated_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()

#经过上面的训练得到最终的模型,不做交叉了直接用全量的数据
model = build_model()
model.fit(train_data, train_targets, epochs=130, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mse_score, test_mae_score)
#测试数据
predictions = model.predict(test_data)
print(predictions[0])
print("这个值经过标准化变化的，没有参考价值" + str(test_targets[0]))