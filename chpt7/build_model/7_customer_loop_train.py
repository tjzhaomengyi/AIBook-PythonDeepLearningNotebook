# -*- coding: utf-8 -*-
__author__ = 'Mike'
import tensorflow as tf
# from tensorflow import keras
import keras
from keras import layers
from keras.datasets import mnist

tf.enable_eager_execution()
#自定义梯度训练，注意apply_gradient要使用tf.keras.optimizer下面的优化器
def train_step(model,inputs, targets):
    with tf.GradientTape() as tape: #loss对weight的一阶导数
        predictions = model(inputs) #只训练那些能训练的参数，不可训练的权重不管，比如该层处理多少批量training=True
        loss = loss_fn(targets, predictions)
    graidents = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(graidents,model.trainable_weights))
    #跟踪指标
    logs = { }
    for metric in metrics:
        metric.update_state(targets, predictions) #每轮对
        logs[metric.name] = metric.result()

    #跟踪损失值
    loss_tracking_metric.update_state(loss)
    logs["loss"] = loss_tracking_metric.result()
    return logs

#重置指标
def reset_metrics():
    for metric in metrics:
        metric.reset_states()
    loss_tracking_metric.reset_states()

#编写循环评估
@tf.function #将tensorflow编译成图，加快计算速度
def test_step(inputs, targets):
    predictions = model(inputs)
    loss = loss_fn(targets, predictions)

    logs={}
    for metric in metrics:
        metric.update_state(targets, predictions)
        logs["val_" + metric.name] = metric.result()

    loss_tracking_metric.update_state(loss)
    logs["val_loss"] = loss_tracking_metric.result()
    return logs


def get_mnist_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    # model = NaiveSequential([NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
    #                          NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax)])
    return model

#自定义指标评估
metric = keras.metrics.SparseCategoricalAccuracy()
targets = [0, 1, 2]
predictions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
metric.update_state(targets, predictions)
current_result = metric.result()
# print(f"result: {current_result:.2f}")

#跟踪标量值
values = [0, 1, 2, 3, 4]
mean_tracker = keras.metrics.Mean() #跟踪器，这个东西挺牛逼
for value in values:
    res = mean_tracker.update_state(value)
    print(mean_tracker.result()) #这个东西非常巧妙就是一个移动平均值，非常平滑
# print(f"Mean of values: {mean_tracker.result():.2f}")

model = get_mnist_model()
loss_fn = keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.RMSprop(1e-3) #注意：如果使用apply_gradients()，这例的optimizer一定要用tf.keras.optimizer,不要使用keras.optimizer
#下面两个记录每轮的更新值
metrics = [keras.metrics.SparseCategoricalAccuracy()]
loss_tracking_metric = keras.metrics.Mean() #准备mean指标跟踪器来跟踪损失就那只

(images, labels), (test_images, test_labels) = mnist.load_data()
print(labels[:10])
images = images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]
training_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
training_dataset = training_dataset.batch(32)
epochs = 3
for epoch in range(epochs):
    reset_metrics()
    for inputs_batch, targets_batch in training_dataset:
        logs = train_step(model, inputs_batch, targets_batch)
    print(f"Results at end of epoch {epoch}")
    for key, value in logs.items():
        print(f"...{key} : {value:.4f}")

#测试验证结果
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
val_dataset = val_dataset.batch(32)
reset_metrics()
for inputs_batch, targets_batch in val_dataset:
    logs = test_step(inputs_batch, targets_batch)
print("Evaluation results:")
for key, value in logs.items():
    print(f"...{key}: {value:.4f}")