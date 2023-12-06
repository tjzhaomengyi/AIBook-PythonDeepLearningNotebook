# -*- coding: utf-8 -*-
__author__ = 'Mike'
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy() #创建分布式对象
print(f"number of devices:{strategy.num_replicas_in_sync}")
with strategy.scope():
    model = get_compiled_model()
model.fit(train_dataset,epochs=100, validation_data=val_dataset, callbacks=callbacks)