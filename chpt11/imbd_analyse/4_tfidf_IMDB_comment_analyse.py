# -*- coding: utf-8 -*-
__author__ = 'Mike'
import os, pathlib, shutil, random
import tensorflow.keras as keras
from keras.layers import TextVectorization
from tensorflow.keras import layers


#构建函数
def get_model(max_tokens=20000, hidden_dim=16):
    inputs = keras.Input(shape=(max_tokens, ))
    x = layers.Dense(hidden_dim, activation="relu")(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    return model


'''从IMDB中分析影评情感'''
imdb_path = "/home/zhaomengyi/Projects/Datas/IMDB/aclImdb_v1/aclImdb/"
base_dir = pathlib.Path(imdb_path)

batch_size = 32
#运行下面一行代码输出应该是Found 20000 files belonging to 2 classes
train_ds = keras.utils.text_dataset_from_directory(base_dir / "train", batch_size=batch_size)
val_ds = keras.utils.text_dataset_from_directory(base_dir / "val", batch_size=batch_size)
test_ds = keras.utils.text_dataset_from_directory(base_dir / "test", batch_size=batch_size)
#将output_mode修改为count，
text_vectorization = TextVectorization(ngrams=2, max_tokens=20000, output_mode="tf_idf")
text_only_train_ds = train_ds.map(lambda x, y: x) #每个train_ds数据只要input，不要target
text_vectorization.adapt(text_only_train_ds) #对训练数据建立词索引
binary_2gram_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
binary_2gram_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
binary_2gram_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
model = get_model()
print(model.summary())
callbacks = [keras.callbacks.ModelCheckpoint("binary_tfidf.keras", save_best_only=True)]
model.fit(binary_2gram_train_ds.cache(), validation_data=binary_2gram_val_ds.cache(), epochs=10, callbacks=callbacks)
model = keras.models.load_model("binary_tfidf.keras")
print(f"Test acc: {model.evaluate(binary_2gram_test_ds)[1]:.3f}")# 0.893

