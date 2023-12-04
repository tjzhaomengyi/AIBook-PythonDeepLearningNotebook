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
for inputs, targets in train_ds:
    print("inputs.shape:", inputs.shape)  #(32,)
    print("inputs.dtype:", inputs.dtype)  #string
    print("targets.shape:", targets.shape)
    print("targets.dtype:", targets.dtype)
    print("inputs[0]:", inputs[0])
    print("targets[0]", targets[0])
    break

'''
方法1：将单词作为集合处理：词袋方法
'''
#生成单词的multi-hot编码的二进制词向量
# 将词表限制为前20000个最常出现的单词，否则，我们需要对训练数据中的每个单词建立索引，可能会有上万个单词只出现一次，因此没有信息量。
#一般来说，20000是用于文本分类的合适的词表大小
text_vectorization = TextVectorization(max_tokens=20000, output_mode="multi_hot")
text_only_train_ds = train_ds.map(lambda x, y: x) #每个train_ds数据只要input，不要target
text_vectorization.adapt(text_only_train_ds) #对训练数据建立词索引
#分别对训练、验证和测试数据进行相同处理，一定要指定num_parallel_cells，以便利用多个CPU核
binary_1gram_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
binary_1gram_val_ds = val_ds.map(lambda x, y : (text_vectorization(x), y), num_parallel_calls=4)
binary_1gram_test_ds = test_ds.map(lambda x, y : (text_vectorization(x), y), num_parallel_calls=4)
#查看结果
for inputs, targets in binary_1gram_train_ds:
    print("inputs.shape:", inputs.shape) #inputs.shape: (32, 20000) 输入每个batch有32个数据，每条数据是一个20000向量
    print("inputs.dtype:", inputs.dtype)
    print("targets.shape:", targets.shape)#targets.shape: (32,)
    print("targets.dtype:", targets.dtype)# tf.Tensor([1. 1. 1. ... 0. 0. 0.], shape=(20000,), dtype=float32)
    print("inputs[0]:", inputs[0])
    print("targets[0]:",targets[0])
    break

#对一元语法二进制模型进行训练和测试
#注意：对数据集调用cache，将其缓存在内存中：利用这种方法，我们只需要在第一轮做一次预处理，在后续轮次可以服用预处理文本！不过只有数据足够小才可以这么装入内存
model = get_model()
print(model.summary())
callbacks = [keras.callbacks.ModelCheckpoint("binary_1gram.keras", save_best_only=True)]
model.fit(binary_1gram_train_ds.cache(), validation_data=binary_1gram_val_ds.cache(), epochs=10, callbacks=callbacks)
model = keras.models.load_model("binary_1gram.keras")
print(f"Test acc:{model.evaluate(binary_1gram_test_ds)[1]:.3f}")#Test acc:0.886

'''

'''
