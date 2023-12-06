# -*- coding: utf-8 -*-
__author__ = 'Mike'
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
print("Device:",tpu.master())

strategy = tf.distribute.TPUStrategy(tpu)
print(f"number of replicas:{strategy.num_replicas_in_sync}")

def build_model(input_size):
    inputs = keras.Input((input_size, input_size, 3))
    x = keras.applications.resnet.preprocess_input(inputs)
    x = keras.applications.resnet.ResNet50(weights=None, include_top=False, pooling="max")(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

with strategy.scope():
    model = build_model(input_size=32)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
model.fit(x_train, y_train, batch_size=1024)
