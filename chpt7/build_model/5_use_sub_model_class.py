# -*- coding: utf-8 -*-
__author__ = 'Mike'
import keras
from keras import layers

# class Classifier(keras.Model):
#     def __init__(self, num_classes=2):
#         super().__init__()
#         if(num_classes == 2):
#             num_units = 1
#             activation = "sigmoid"
#         else:
#             num_units = num_classes
#             activation = "softmax"
#         self.dense = layers.Dense(num_units, activation=activation)
#
#     def call(self, inputs):
#         return self.dense(inputs)
#
# inputs = keras.Input(shape=(3,))
# features = layers.Dense(64, activation="relu")(inputs)
# outputs = Classifier(num_classes=10)(features)
# model = keras.Model(inputs=inputs, outputs=outputs)
class Classifier(keras.Model):

    def __init__(self, num_classes=2):
        super().__init__()
        if num_classes == 2:
            num_units = 1
            activation = "sigmoid"
        else:
            num_units = num_classes
            activation = "softmax"
        self.dense = layers.Dense(num_units, activation=activation)

    def __call__(self, inputs):
        return self.dense(inputs)

inputs = keras.Input(shape=(3,))
features = layers.Dense(64, activation="relu")(inputs)
outputs = Classifier(num_classes=10)(features)
model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())