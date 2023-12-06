# -*- coding: utf-8 -*-
__author__ = 'Mike'
import random
import tensorflow as tf
import string
import re
import tensorflow.keras as keras
import keras.layers as layers

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


def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization([decoded_sentence])
        next_token_predictions = seq2seq_rnn.predict([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(next_token_predictions[0, i, :])
        sampled_token = spar_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence

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
strip_chars = string.punctuation + "¿"
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

#准备翻译任务的数据
batch_size = 64
train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)
for inputs, targets  in train_ds.take(1):
    print(f"inputs['english'].shape:{inputs['english'].shape}")
    print(f"inputs['spanish'].shape:{inputs['spanish'].shape}")
    print(f"targets.shape:{targets.shape}")
# inputs['english'].shape:(64, 20)
# inputs['spanish'].shape:(64, 20)
# targets.shape:(64, 20)

'''-------------------RNN的Seq2Seq模型-----------------------------'''
# inputs  = keras.Input(shape=(sequence_length,), dtype="int64")
# x = layers.Embedding(input_dim=vocab_size, output_dim=128)(inputs)
# x = layers.LSTM(32, return_sequences=True)(x)
# outputs = layers.Dense(vocab_size, activation="softmax")(x)
# model = keras.Model(inputs, outputs)
'''-----------使用GRU代替LSTM，因为GRU只有一个状态向量，LSTM有多个'''
embed_dim = 256
latent_dim = 1024
#1、基于GRU的编码器
source = keras.Input(shape=(None, ), dtype="int64", name="english")
x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(source)
#编码后的源句子即为双向GRU的最后一个输出
encoded_source = layers.Bidirectional(layers.GRU(latent_dim), merge_mode="sum")(x)
#2、添加GRU的解码器，基于GRU的解码器语端到端模型
past_target = keras.Input(shape=(None,), dtype="int64", name="spanish")
x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(past_target)
decoder_gru = layers.GRU(latent_dim, return_sequences=True)
x = decoder_gru(x, initial_state=encoded_source)
x = layers.Dropout(0.5)(x)
target_next_step = layers.Dense(vocab_size, activation="softmax")(x)
seq2seq_rnn = keras.Model([source, past_target], target_next_step)
seq2seq_rnn.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
callbacks = [keras.callbacks.ModelCheckpoint("translate_seq_to_seq.keras", save_best_only=True)]
seq2seq_rnn.fit(train_ds, epochs=15, validation_data=val_ds, callbacks=callbacks)
#训练结果： 893s 686ms/step - loss: 0.9592 - accuracy: 0.6956 - val_loss: 1.0560 - val_accuracy: 0.6417


#利用RNN编码器和RNN解码器来翻译句子
spa_vocab = target_vectorization.get_vocabulary()
spar_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20
test_eng_texts = [pair[0] for pair in test_pairs]
for _ in range(20):
    input_sentence = random.choice(test_eng_texts)
    print("-")
    print(input_sentence)
    print(decode_sequence(input_sentence))