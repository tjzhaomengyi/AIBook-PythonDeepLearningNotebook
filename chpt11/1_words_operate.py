# -*- coding: utf-8 -*-
__author__ = 'Mike'
# from tensorflow.keras.layers import TextVectorization
from keras.layers import TextVectorization

'''默认情况下TextVectorization使用转换为小写字母并删除标点符号，词元化的方法是利用空格拆分
注意：TextVectorization是
'''
text_vectorizaiton = TextVectorization(output_mode="int") #设置该层的返回值为整数索引的单词序列
#生成词元列表
dataset = ["I write, erase, rewrite", "Erase again, and then", "A poppy blooms"]
text_vectorizaiton.adapt(dataset)
print(text_vectorizaiton.get_vocabulary())

'''下面例子对单词进行编码，然后再反编码回单词'''
vocabulary = text_vectorizaiton.get_vocabulary()
test_sentence = "I write, rewrite, and still rewrite again"
encoded_sentence = text_vectorizaiton(test_sentence)
print(encoded_sentence)
inverse_vocab = dict(enumerate(vocabulary))
print(inverse_vocab)
decoded_sentence =   " ".join([inverse_vocab[int(i)] for i in encoded_sentence])
print(decoded_sentence)

