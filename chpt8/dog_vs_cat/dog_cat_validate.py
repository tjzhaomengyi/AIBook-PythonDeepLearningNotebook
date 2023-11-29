# -*- coding: utf-8 -*-
__author__ = 'Mike'
import keras
from keras import layers
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os, shutil, pathlib
import matplotlib.pyplot as plt

new_base = "/home/zhaomengyi/Projects/AIProjects/Book_PythonDeepLearn/chpt8/datas"
new_base_dir = pathlib.Path(new_base)
datagen = ImageDataGenerator()
test_dataset = datagen.flow_from_directory(new_base_dir / "test", target_size=(180, 180), batch_size=32, class_mode="binary")
test_model = keras.models.load_model("convnet_from_scratch.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy:{test_acc:.3f}")