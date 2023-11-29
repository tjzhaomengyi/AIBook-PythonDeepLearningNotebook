# -*- coding: utf-8 -*-
__author__ = 'Mike'
import keras
from keras import layers
import numpy as np

vocabulary_size = 10000
num_tags = 100
num_departments = 4

#多个输入
title = keras.Input(shape=(vocabulary_size, ), name="title")
text_body = keras.Input(shape=(vocabulary_size, ), name="text_body")
tags = keras.Input(shape=(num_tags, ), name="tags")

#将输入拼接成featurs,使用api的方式
features = layers.Concatenate()([title, text_body, tags])
features = layers.Dense(64, activation="relu")(features)

#定义多个输出
priority = layers.Dense(1, activation="sigmoid", name="priority")(features)
department = layers.Dense(num_departments, activation="softmax", name="department")(features)

model = keras.Model(inputs=[title, text_body, tags], outputs=[priority, department])

#训练模型
num_samples = 1280
title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(0, 3, size=(num_samples, num_departments))

''''
#方法1：通过列表的方法给出训练参数
model.compile(optimizer="rmsprop", loss=["mean_squared_error", "categorical_crossentropy"], metrics=[["mean_absolute_error"], ["accuracy"]])
model.fit([title_data, text_body_data, tags_data],[priority_data, department_data], epochs=1)
model.evaluate([title_data, text_body_data, tags_data], [priority_data, department_data])
priority_preds, department_preds = model.predict([title_data, text_body_data, tags_data])
'''
#方法2：通过字典方式
model.compile(optimizer="rmsprop", loss={"priority":"mean_squared_error", "department":"categorical_crossentropy"},
              metrics={"priority":["mean_absolute_error"], "department":["accuracy"]})
model.fit({"title": title_data, "text_body":text_body_data, "tags":tags_data},
          {"priority":priority_data, "department":department_data},
          epochs=1)
model.evaluate({"title": title_data, "text_body":text_body_data, "tags":tags_data},
               {"priority":priority_data, "department":department_data})
priority_preds, department_preds = model.predict({"title":title_data, "text_body":text_body_data, "tags":tags_data})

'''
#绘制模型结构
keras.utils.plot_model(model, "ticket_classifier.png")
keras.utils.plot_model(model,"text_classifier_shape_info.png",show_shapes=True)
'''
print(model.layers)
print(model.layers[3].output)

#迁移中间层，把中间的第四层抽出来再生成一个输出结果
features = model.layers[4].output
difficulty = layers.Dense(3, activation="softmax", name="difficulty")(features)
new_model = keras.Model(inputs=[title, text_body, tags], outputs=[priority, department, difficulty])
keras.utils.plot_model(new_model, "update_ticket_classifier.png", show_shapes=True)