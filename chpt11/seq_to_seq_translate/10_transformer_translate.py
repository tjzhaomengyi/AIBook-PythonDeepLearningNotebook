# -*- coding: utf-8 -*-
__author__ = 'Mike'
import random
import tensorflow as tf
import string
import re
import tensorflow.keras as keras
import keras.layers as layers
import numpy as np


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


'''根据书中给的Decoder解码器图，需要两个Attention层'''
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim)])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        # 这个属性可以确保该层将输入掩码传递给输出。Keras中的掩码是可选项。如果一个层没有实现compute_mask()，并且没有暴露supports_masking属性，
        # 那么向该层传入掩码就会报错
        self.suppoorts_masking=True

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim
        })
        return config

    #为了防止模型关注未来信息，在生成N+1个目标词元时，应该仅使用目标序列中0~N个词元的信息，下面这个代码就是这个作用
    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32") #生成形状(sequence_length, sequence_length)的矩阵，其中一半为0， 一半为1
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat([
            tf.expand_dims(batch_size, -1),
            tf.constant([1, 1], dtype=tf.int32)
        ], axis=0)
        return tf.tile(mask, mult)

    #前向传播
    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)#获取因果掩码
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")#准备输入掩码，描述目标序列的填充位置
            padding_mask = tf.minimum(padding_mask, causal_mask) #将两个掩码合并
        attention_output_1 = self.attention_1(query=inputs, value=inputs, key=inputs, attention_mask=causal_mask)
        attention_output_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(query=attention_output_1, value=encoder_outputs, key=encoder_outputs, attention_mask=padding_mask)
        attention_output_2 = self.layernorm_2(attention_output_1 + attention_output_2)
        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm_3(attention_output_2 + proj_output)



def custom_standardization(input_string):
    lowcase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowcase, f"[{re.escape(strip_chars)}]", "")

#准备翻译任务的数据集,单个任务集处理
def format_dataset(eng, spa):
    eng = source_vectorization(eng)
    spa = target_vectorization(spa)
    return ({"english": eng,"spanish":spa[:, :-1]}, #输入西班牙句子不包含最后一个词元，以保证输入和目标具有相同长度
            spa[:, 1:]) #目标句子向后偏移一个时间步，二者长度相同，都是20个单词

def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=4)
    return dataset.shuffle(2048).prefetch(16).cache() #利用内存缓存加快速度


def decode_sequcene(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization([decoded_sentence])[:,:-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " "+ sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence




embed_dim = 256
dense_dim = 2048
num_heads = 8
sequence_length = 600
max_length = 600
max_tokens = 20000
vocab_size = 20000
embed_dim = 256
num_heads = 2
dense_dim = 32



text_file = "/home/zhaomengyi/Projects/Datas/Translate/spa-eng/spa.txt"
with open(text_file) as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []
for line in lines:
    english, spanish = line.split("\t")
    spanish = "[start] " + spanish + " [end]" #将start和end分别添加到西班牙语句子的开头和结尾
    text_pairs.append((english, spanish))

print(random.choice(text_pairs)) #('Correct the following sentences.', '[start] Corrige las siguientes frases. [end]')

#将训练集和测试集打乱
random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples:num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples:]

'''注意：在真实的翻译中都要保留标点，这里为了简单删除标点'''
#为西班牙语的TextVecotrization层准备一个自定义的字符串标准化函数：保留[和]，去掉¿和其他标点
strip_chars = string.punctuation #线上暂时删除+ "¿"
strip_chars = strip_chars.replace("[","")
strip_chars = strip_chars.replace("]","")



#只查看每种语言前15000个最常见的单词，并将句子长度限制为20个单词
vocab_size = 15000
sequence_length = 20

#英语层
source_vectorization = layers.TextVectorization(max_tokens=vocab_size, output_mode="int", output_sequence_length=sequence_length)
#西班牙语层，生成的西班牙语句子多了一个词元，因为在训练过程中需要将句子偏移一个时间步！【注意：这里其实就是表示用当前的英语要预测下一个位置的西班牙语】
target_vectorization = layers.TextVectorization(max_tokens=vocab_size, output_mode="int", output_sequence_length=sequence_length+1,
                                                standardize=custom_standardization)
#生成编码器
train_english_texts = [pair[0] for pair in train_pairs]
train_spanish_texts = [pair[1] for pair in train_pairs]
source_vectorization.adapt(train_english_texts)
target_vectorization.adapt(train_spanish_texts)


spa_vocab = target_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20


#准备翻译任务的数据
batch_size = 64
train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)
#注意：翻译要用的是Decoder和Encoder的Transformer，Encoder输入待翻译训练句子，Decoder输入翻译结果的训练句子
encoder_inputs = keras.Input(shape=(None, ), dtype="int64", name="english") #训练数据中的待翻译句子，给Encoder
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="spanish") #训练数据中翻译结果的句子给Decoder
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs) #注意：模型输入要把encoder和decoder的输入都加上！！！！
transformer.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
callbacks = [keras.callbacks.ModelCheckpoint("translate_Transformer.keras", save_best_only=True)]
transformer.fit(train_ds, epochs=30, validation_data=val_ds, callbacks=callbacks)

#训练Epoch 30/30 1302/1302 [==============================] - 228s 175ms/step - loss: 0.9440 - accuracy: 0.8766 - val_loss: 1.1505 - val_accuracy: 0.8423

''''--------执行翻译任务------'''
transformer = keras.models.load_model("translate_Transformer.keras", custom_objects={"TransformerEncoder": TransformerEncoder,
                                                                                  "PositionalEmbedding": PositionalEmbedding,
                                                                                   "TransformerDecoder": TransformerDecoder })
test_eng_texts = [pair[0] for pair in test_pairs]
for _ in range(20):
    input_sentence = random.choice(test_eng_texts)
    print("------")
    print(input_sentence)
    print(decode_sequcene(input_sentence))

'''
------
School begins at nine and is over at six.
[start] en malo y bien [end]
------
Tom laughed at himself.
[start] tom se fue a la necesito [end]
------
Keep smiling.
[start] parece un tuve [end]
------
Do you want some beer?
[start] gustaría [UNK] está solo comprar [end]
------
He told them that he had had a wonderful time.
[start] los es no [UNK] algo lo que él es tenía [end]
------
The war had united the American people.
[start] el dos algo lo que le vino [end]
------
We live ten minutes away from him.
[start] lo más mundo nunca [UNK] [end]
------
I didn't hear it.
[start] no lo quiere [end]
------
I think I've been here before.
[start] trabajo que muchos [end]
------
I don't know if we want to do that.
[start] no nos está lo mes hacer [end]
------
Let me see the pictures you took in Paris.
[start] le hice a sueño a el tuve [end]
------
He has a lot more money than I have.
[start] los tiene una ahora para bastante en cosas me tiene [end]
------
Tom was fast asleep.
[start] tom fue la les [end]
------
That was a dumb question.
[start] fue un [UNK] en nadar [end]
------
I wish to talk to your daughter.
[start] yo le [UNK] a tu o [end]
------
Tom got out of the taxi.
[start] tom se [UNK] [end]
------
If you make new friends, don't forget the old ones.
[start] no verdad [UNK] a el sabe de tom tres mejor [end]
------
Is Tom in on this?
[start] pasó tom [end]
------
What was the last concert you went to?
[start] gustaría estaba de esto de [UNK] [end]
------
Don't get yourself involved in that.
[start] no te [UNK] eres la gato [end]

'''
