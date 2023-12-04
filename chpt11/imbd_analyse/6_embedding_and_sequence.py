# -*- coding: utf-8 -*-
__author__ = 'Mike'
import tensorflow.keras as keras
from keras import layers
import tensorflow as tf
import pathlib

imdb_path = "/home/zhaomengyi/Projects/Datas/IMDB/aclImdb_v1/aclImdb/"
base_dir = pathlib.Path(imdb_path)
batch_size = 32
#运行下面一行代码输出应该是Found 20000 files belonging to 2 classes
train_ds = keras.utils.text_dataset_from_directory(base_dir / "train", batch_size=batch_size)
val_ds = keras.utils.text_dataset_from_directory(base_dir / "val", batch_size=batch_size)
test_ds = keras.utils.text_dataset_from_directory(base_dir / "test", batch_size=batch_size)

max_length = 600
max_tokens = 20000
#为保持输入大小可控，我们在前600个单词处截断输入。这是一个合理的选择，因为评论的平均长度是233个单词，只有5%的评论超过600单词
text_vetorization = layers.TextVectorization(max_tokens=max_tokens, output_mode="int", output_sequence_length=max_length)
text_only_train_ds = train_ds.map(lambda x, y: x) #每个train_ds数据只要input，不要target
text_vetorization.adapt(text_only_train_ds)
int_train_ds = train_ds.map(lambda x, y:(text_vetorization(x), y), num_parallel_calls=4)
int_val_ds = val_ds.map(lambda x, y:(text_vetorization(x), y), num_parallel_calls=4)
int_test_ds = test_ds.map(lambda x, y:(text_vetorization(x), y), num_parallel_calls=4)

'''
下面创建模型，将整数序列转换为向量序列，对整数进行one-hot编码，每个维度代表一个单词。在这些one-hot向量上，添加一个简单的双向LSTM

'''
inputs = keras.Input(shape=(None,), dtype="int64") #每个输入是一个整数序列,这里应该是一个维度最大为600的向量
#embedded = tf.one_hot(inputs, depth=max_tokens) #将整数编码为20000维度的二进制向量，简单的20000 embedding向量,这个他妈的太大了
#embedding层的输入形状是(batch_size,sequence_length)的2阶整数张量，其中每个元素都是一个整数序列。
# 该层返回是一个形状为（batch, sequence_length,embedding_dimessionality）的三阶浮点数张量。
embedding_layer = layers.Embedding(input_dim=max_tokens, output_dim=256)(inputs)
#带有掩码的embedding
embedding_layer = layers.Embedding(input_dim=max_tokens, output_dim=256, mask_zero=True)(inputs)
x = layers.Bidirectional(layers.LSTM(32))(embedding_layer) #添加一个双向LSTM
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
callbacks = [keras.callbacks.ModelCheckpoint("one_hot_bidir_gru.keras", save_best_only=True)]
model.fit(int_train_ds, validation_data=int_val_ds, epochs=10, callbacks=callbacks)
model = keras.models.load_model("one_hot_bidir_gru.keras")
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}") #Test acc: 0.861
#测试结果也不太好，只有87%，还没有2gram那个模型好，因为2gram模型处理的是完整的评论，而这个序列模型在600个单词之后就截断序列。
#并且使用双向RNN，一个正序处理词元，另一个逆序处理词元，正序的在600个单词后只看到填充只。如果句子很短，RNN传播的信息逐渐消失。
#这里就要使用掩码masking，掩码是有1和0组成的张量，形状为(batch_size,sequence_length)，其中元素mask[i,t]表示第i个样本第t时间步是否应该被跳过，默认是不开的
