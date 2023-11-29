# -*- coding: utf-8 -*-
__author__ = 'Mike'
import keras
from keras.datasets import mnist
from keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt

# tf.enable_eager_execution()
def get_mnist_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    return model

'''
自定义指标,需要编写状态更新逻辑，由update_state()实现
'''
class RootMeanSquaredError(keras.metrics.Metric):
    def __init__(self, name="rmse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mse_sum = self.add_weight(name="mse_sum", initializer="zeros")
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros", dtype="int32")

    def update_state(self, y_true, y_pred, sample_weight=None): #这个例子又不是很好，这个one_hot做的太差了
        y_true = tf.one_hot(tf.cast(y_true, dtype=tf.int32), depth=tf.shape(y_pred)[1]) #为了匹配MNIST模型，我们需要分类预测值与整数标签
        mse = tf.reduce_sum(tf.square(y_true - y_pred))
        self.mse_sum.assign_add(mse)
        num_samples = tf.shape(y_pred)[0]
        self.total_samples.assign_add(num_samples)
    #返回当前指标
    def result(self):
        return tf.sqrt(self.mse_sum / tf.cast(self.total_samples, tf.float32))
    #清空轮次指标
    def reset_states(self):
        self.mse_sum.assign(0.)
        self.total_samples.assign(0)
'''
class RootMeanSquaredError(keras.metrics.Metric):

    def __init__(self, name="rmse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mse_sum = self.add_weight(name="mse_sum", initializer="zeros")
        self.total_samples = self.add_weight(
            name="total_samples", initializer="zeros", dtype="int32")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.one_hot(tf.cast(y_true, dtype=tf.int32), depth=tf.shape(y_pred)[1])
        mse = tf.reduce_sum(tf.square(y_true - y_pred))
        self.mse_sum.assign_add(mse)
        num_samples = tf.shape(y_pred)[0]
        self.total_samples.assign_add(num_samples)

    def result(self):
        return tf.sqrt(self.mse_sum / tf.cast(self.total_samples, tf.float32))

    def reset_state(self):
        self.mse_sum.assign(0.)
        self.total_samples.assign(0)
'''

'''
自定义回调函数
'''
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.per_batch_losses = []

    def on_batch_end(self, batch, logs):
        self.per_batch_losses.append(logs.get("loss"))

    def on_epoch_end(self, epoch, logs):
        plt.clf()
        plt.plot(range(len(self.per_batch_losses)), self.per_batch_losses, label="Training loss for each batch")
        plt.xlabel(f"batch (epoch {epoch})")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"plot_at_epoch_{epoch}")
        self.per_batch_losses = []





(images, labels), (test_images, test_labels) = mnist.load_data()
print(labels[:10])
images = images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]


# y_true = tf.one_hot(tf.cast(train_labels, dtype=tf.int32), depth=10)
# print(y_true)
model = get_mnist_model()
#自定义回调函数,第一个回调如果val_accuracy得不到改善，那么就终止训练，patience表示如果在2轮得不到改善就停止
#第二个回调，每轮保存一次模型,monitor表示只有当val_loss得到改善才会保存模型
callbacks_list = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2),
    keras.callbacks.ModelCheckpoint(filepath="checkpoint_path.keras", monitor="val_loss", save_best_only=True)
]
#添加TensorBoard
tensorboard = keras.callbacks.TensorBoard(log_dir='./tensorboard')

model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy", RootMeanSquaredError()])
model.fit(train_images, train_labels, epochs=3, validation_data=(val_images, val_labels), callbacks=[tensorboard]) #callbacks_list,LossHistory()
predictions = model.predict(test_images)
print(predictions[0])