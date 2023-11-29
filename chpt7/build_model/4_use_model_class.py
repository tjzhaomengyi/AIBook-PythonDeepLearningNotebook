# -*- coding: utf-8 -*-
__author__ = 'Mike'
import keras
from keras import layers
import numpy as np

'''
子模型的方式构建模型最灵活,这个例子不好
'''

class CustomerTicketModel(keras.Model):
    def __init__(self, num_departments):
        super().__init__()
        #借助python的灵活性，下面定义的全是api方式构建各个layer层的方法
        self.concat_layer = layers.Concatenate()
        self.mixing_layer = layers.Dense(64, activation="relu")
        self.priority_scorer = layers.Dense(1, activation="sigmoid",name="priority")
        self.department_classifier = layers.Dense(num_departments, activation="softmax", name="department")

    def __call__(self, inputs):
        #call定义前向传播
        title = inputs["title"]
        text_body = inputs["text_body"]
        tags = inputs["tags"]
        #使用init定义的“层方法”构建每层
        features = self.concat_layer([title, text_body, tags])
        features = self.mixing_layer(features)
        priority = self.priority_scorer(features)
        department = self.department_classifier(features)
        model = keras.Model(inputs=[title, text_body, tags], outputs=[priority, department])
        return model#返回模型的两个分类输出结果


#构造数据
vocabulary_size = 10000
num_tags = 100
num_departments = 4
num_samples = 1280
title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(0, 3, size=(num_samples, num_departments))
#注意：这里不能直接使用实际的数组，还是要传一个占位的tensor进去
title = keras.Input(shape=(vocabulary_size, ), name="title")
text_body = keras.Input(shape=(vocabulary_size, ), name="text_body")
tags = keras.Input(shape=(num_tags, ), name="tags")
model = CustomerTicketModel(num_departments=4)(
    {"title":title, "text_body":text_body, "tags":tags}
)
#方法2：通过字典方式
model.compile(optimizer="rmsprop", loss={"priority":"mean_squared_error", "department":"categorical_crossentropy"},
              metrics={"priority":["mean_absolute_error"], "department":["accuracy"]})
model.fit({"title": title_data, "text_body":text_body_data, "tags":tags_data},
          {"priority":priority_data, "department":department_data},
          epochs=1)
model.evaluate({"title": title_data, "text_body":text_body_data, "tags":tags_data},
               {"priority":priority_data, "department":department_data})
priority_preds, department_preds = model.predict({"title":title_data, "text_body":text_body_data, "tags":tags_data})