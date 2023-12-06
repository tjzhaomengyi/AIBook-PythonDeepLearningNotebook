# -*- coding: utf-8 -*-
__author__ = 'Mike'
'''根据IMDB生成影评'''
import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import TextVectorization
import numpy as np
import keras.layers as layers

#增减线上GPU限制
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        # 或者，设置每个 GPU 进程的显存分配比例
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=0.1)]
        )

imdb_path = "/home/zhaomengyi/Projects/Datas/IMDB/aclImdb_v1/aclImdb/"
#imdb_path = "/students/julyedu_693906/Projects/Datas/aclImdb"
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


class TextGenerator(keras.callbacks.Callback):
    # prompt:提示词，作为文本生成的种子； generate_length要生成多少单词，temperatures用于采样的温度值
    def __init__(self, prompt, generate_length, model_input_length, temperatures=(1.,), print_freq=1):
        self.prompt = prompt
        self.generate_length = generate_length
        self.model_input_length = model_input_length
        self.temperatures = temperatures
        self.print_freq = print_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_freq != 0:
            return
        for temperature in self.temperatures:
            print("== Generating with temperature", temperature)
            # 生成文本时，初始文本为提示词
            sentence = self.prompt
            for i in range(self.generate_length):
                # 将当前序列输入模型
                tokenized_sentence = text_vetorization([sentence])
                predictions = self.model(tokenized_sentence)
                # 获取最后一个时间步的预测结果，并利用它来采样一个新词
                next_token = sample_next(predictions[0, i, :])
                sampled_token = tokens_index[next_token]
                sentence += " " + sampled_token  # 将这个新词添加到当前序列中， 并重复上述过程
            print(sentence)


'''-------------------------------其他工具函数------------------------------------------'''
'''softmax温度函数，用来计算下一个取值的惊喜程度，温度越高越惊喜'''
def reweight_distribution(original_distribution, temperature=0.5):
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)


#创建语言模型数据集
def prepare_lm_dataset(text_batch):
    #将字符串文本批量转化为整数序列批量
    vectorized_sequences = text_vetorization(text_batch)
    x = vectorized_sequences[:, :-1] #通过删掉最后一个词来构建输入
    y = vectorized_sequences[:, 1:] #删掉第0位置的词构建输出
    return x, y

#文本生成回调函数,供TextGenerator使用
def sample_next(predictions, temperature=1.0):
    #利用softmax温度函数，创建单词惊喜度
    preditions = np.asarray(predictions).astype("float64")
    preditions = np.log(predictions) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)








dataset = keras.utils.text_dataset_from_directory(directory=imdb_path, label_mode=None, batch_size=256)
dataset = dataset.map(lambda x: tf.strings.regex_replace(x, "<br />"," "))
sequence_length = 100
vocab_size = 15000 #只考虑前15000个最常见的单词，其他设置为[UNK]
#将原始数据编码
text_vetorization = TextVectorization(max_tokens=vocab_size, output_mode="int", output_sequence_length=sequence_length)
text_vetorization.adapt(dataset)
#使用text_vetorization将dataset变成编码
lm_dataset = dataset.map(prepare_lm_dataset, num_parallel_calls=4)

embed_dim = 256
latent_dim = 2048
num_heads = 2

inputs = keras.Input(shape=(None, ), dtype="int64")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
x = TransformerEncoder(embed_dim, latent_dim, num_heads)(x, x)
outputs = layers.Dense(vocab_size, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop")

#一个字典，将单词索引映射为字符窜，可用于文本解码
tokens_index = dict(enumerate(text_vetorization.get_vocabulary()))

# 种子
prompt = "This movie"
text_gen_callback = TextGenerator(prompt, generate_length=50, model_input_length=sequence_length, temperatures=(0.2, 0.5, 0.7, 1.,1.5))
save_callback = [keras.callbacks.ModelCheckpoint("generator_imdb_commments.keras", save_best_only=True)]
model.fit(lm_dataset, epochs=5, callbacks=[text_gen_callback, save_callback])
