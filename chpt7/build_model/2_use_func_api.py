# -*- coding: utf-8 -*-
__author__ = 'Mike'
import keras
from keras import layers

inputs = keras.Input(shape=(3,), name="my_input") #注意这里（3，）其实表示的是n行3列
features = layers.Dense(64, activation="relu")(inputs)
print(features.shape)
outputs = layers.Dense(10, activation="softmax")(features)
model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())