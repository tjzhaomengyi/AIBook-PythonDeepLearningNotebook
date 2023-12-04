# -*- coding: utf-8 -*-
__author__ = 'Mike'
import os, pathlib, shutil, random
import tensorflow.keras as keras
from keras.layers import TextVectorization
from tensorflow.keras import layers
import tensorflow as tf

'''
如何利用训练好的模型，并使用对应词向量化层完成模型预测
'''
#注意这里构建的时候要重新构建
imdb_path = "/home/zhaomengyi/Projects/Datas/IMDB/aclImdb_v1/aclImdb/"
base_dir = pathlib.Path(imdb_path)
batch_size = 32
#运行下面一行代码输出应该是Found 20000 files belonging to 2 classes
train_ds = keras.utils.text_dataset_from_directory(base_dir / "train", batch_size=batch_size)
text_vectorization = TextVectorization(ngrams=2, max_tokens=20000, output_mode="tf_idf")
text_only_train_ds = train_ds.map(lambda x, y: x) #每个train_ds数据只要input，不要target
text_vectorization.adapt(text_only_train_ds) #对训练数据建立词索引

inputs = keras.Input(shape=(1,), dtype="string") #每个输入样本都是一个字符串
processed_inputs = text_vectorization(inputs)
model = keras.models.load_model("binary_tfidf.keras")
outputs = model(processed_inputs)
inference_model = keras.Model(inputs, outputs)

#我们得到的模型可以处理原始字符串组成的批向量
raw_text_data = tf.convert_to_tensor(["That was an excellent movie, i love it"])
predictions = inference_model(raw_text_data)
print(f"{float(predictions[0] * 100):.2f}% positive") #90.02% positive