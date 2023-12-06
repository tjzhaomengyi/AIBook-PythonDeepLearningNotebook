# -*- coding: utf-8 -*-
__author__ = 'Mike'
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import pathlib

'''
#对单词计算相关性向量的伪代码
def sel_attention(input_squence):
    output = np.zeros(shape=input_squence.shape)
    for i, pivot_vector in enumerate(input_squence): #对输入序列中的每个词元进行迭代
        scores = np.zeros(shape=(len(input_squence),))
        for j, vector in enumerate(input_sequence):
            scores[j] = np.dot(pivot_vector, vector.T) #计算词元与其余每个单词之间的点积（注意力分数）
        scores /= np.squrt(input_squence.shape(1)) #本行和下一行利用规范化因子进行缩放
        scores = softmax(scores)
        new_pivot_representation = np.zeros(shape=pivot_vector.shape)
        for j, vector in enumerate(input_squence):
            new_pivot_representation += vector * scores[j] #利用注意力进行加权，对所有词元进行求和
        output[i] = new_pivot_representation #这个总和就是输出
    return output
'''

# num_head = 4
# embed_dim = 256
# mha_layer = layers.MultiHeadAttention(num_heads=num_head, key_dim=embed_dim)
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim #输入词元向量的尺寸
        self.dense_dim = dense_dim #内部密集层的尺寸
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.layernorm_1 = layers.LayerNormalization()
        self.dense_proj = keras.Sequential([layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),])
        '''注意：这里使用的是layerNormalizaiton：batchnormaliztion是从多个样本中收集信息，一获得特征均值和方差。
            LayerNormaliztion使用分别汇聚每个序列中的数据，更适用于序列数据
        '''
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        #Embedding层生成的掩码是二维的，但是注意力层的输入应该是三维或者四维的，所以要增加mask的层数
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs) #, attenion_mask=mask这里有问题
        proj_input = self.layernorm_1(inputs + attention_output) #第一部分的残差理解
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output) #第二部分的残差连接

    def get_config(self): #实现序列化，保存模型
        config = super().get_config()
        config.update({"embed_dim":self.embed_dim, "num_heads":self.num_heads, "dense_dim":self.dense_dim})
        return config

#词位置嵌入模型
class PositionalEmbedding(layers.Layer):#位置嵌入的缺点是要事先知道序列长度
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        #准备一个embedding层用于保存词元索引
        self.token_embedding = layers.Embedding(input_dim=input_dim, output_dim=output_dim)
        self.position_embedding = layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embedding(inputs)
        embedded_positions = self.position_embedding(positions)
        return embedded_tokens + embedded_tokens

    #与Embedding层一样，该层应该能够生成掩码，从而可以忽略输入填充中的0.框架会自动调用compute_mask方法，并将掩码传递给下一层
    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    #实现序列化以便保存模型
    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config


vocab_size = 20000
embed_dim = 256
num_heads = 2
dense_dim = 32

imdb_path = "/home/zhaomengyi/Projects/Datas/IMDB/aclImdb_v1/aclImdb/"
# imdb_path = "/students/julyedu_693906/Projects/Datas/aclImdb"
base_dir = pathlib.Path(imdb_path)
batch_size = 32
#运行下面一行代码输出应该是Found 20000 files belonging to 2 classes
train_ds = keras.utils.text_dataset_from_directory(base_dir / "train", batch_size=batch_size)
val_ds = keras.utils.text_dataset_from_directory(base_dir / "val", batch_size=batch_size)
test_ds = keras.utils.text_dataset_from_directory(base_dir / "test", batch_size=batch_size)

sequence_length = 600
max_length = 600
max_tokens = 20000
#为保持输入大小可控，我们在前600个单词处截断输入。这是一个合理的选择，因为评论的平均长度是233个单词，只有5%的评论超过600单词
text_vetorization = layers.TextVectorization(max_tokens=max_tokens, output_mode="int", output_sequence_length=max_length)
text_only_train_ds = train_ds.map(lambda x, y: x) #每个train_ds数据只要input，不要target
text_vetorization.adapt(text_only_train_ds)
int_train_ds = train_ds.map(lambda x, y:(text_vetorization(x), y), num_parallel_calls=4)
int_val_ds = val_ds.map(lambda x, y:(text_vetorization(x), y), num_parallel_calls=4)
int_test_ds = test_ds.map(lambda x, y:(text_vetorization(x), y), num_parallel_calls=4)

inputs = keras.Input(shape=(None, ), dtype="int64")
# x = layers.Embedding(vocab_size, embed_dim)(inputs)
# 替换成位置嵌入模型
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
#TransformerEncoder返回的是完整的序列，所以我们需要全局汇聚层将每个序列转化为单个向量，以便进行分类
x = layers.GlobalMaxPool1D()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
print(model.summary())

callbacks = [keras.callbacks.ModelCheckpoint("full_transformer_encoder.keras", save_best_only=True)]
model.fit(int_train_ds, validation_data=int_val_ds, epochs=20, callbacks=callbacks)
#【注意】有自定义模型输入输入的时候一定要带着自定义模型的名字和序列化类
model = keras.models.load_model("full_transformer_encoder.keras", custom_objects={"TransformerEncoder": TransformerEncoder,
                                                                                  "PositionalEmbedding": PositionalEmbedding})
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}") #不带位置编码的结果0.879 ,带编码位置的是0.883

