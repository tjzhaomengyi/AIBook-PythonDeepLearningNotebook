# -*- coding: utf-8 -*-
__author__ = 'Mike'
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()


class TimeSeriesOperator():
    def __init__(self):
        return


    def create_time_series_dataset(data, sequence_length, batch_size, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices(data) #加载np数据，避免将整个数据集加载到内存中
        dataset = dataset.window(size=sequence_length + 1, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(sequence_length + 1))
        dataset = dataset.map(lambda window: (window[:-1], window[-1]))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(data) - sequence_length)
        dataset = dataset.batch(batch_size)
        return dataset


    #带sample采样的时间序列生成
    def create_time_series_dataset_with_sample(data, sequence_length, batch_size, sampling_rate=1, shuffle=True):
        # Create a dataset from the input data
        dataset = tf.data.Dataset.from_tensor_slices(data)
        # Create windows as before
        dataset = dataset.window(size=sequence_length + 1, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(sequence_length + 1))
        dataset = dataset.map(lambda window: (window[:-1], window[-1]))
        # Sample the dataset based on the provided sampling_rate
        if sampling_rate > 1:
            dataset = dataset.enumerate()
            dataset = dataset.filter(lambda i, _: i % sampling_rate == 0)
            dataset = dataset.map(lambda _, data: data)
        # Shuffle and batch the dataset
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(data) - sequence_length)
        dataset = dataset.batch(batch_size)

        return dataset

# Example data
data = np.arange(96)
# Hyperparameters
sequence_length = 3
batch_size = 2
sampling_rate = 3  # Set the desired sampling rate
# Create the time series dataset
dataset = TimeSeriesOperator.create_time_series_dataset_with_sample(data, sequence_length, batch_size, sampling_rate=sampling_rate, shuffle=True)
# Print the batches
print(tf.data.experimental.cardinality(dataset))
for batch in dataset:
    print("Input (X):", batch[0].numpy())
    print("Target (y):", batch[1].numpy())
    print("------------------------------")
#
# # 示例数据
# data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
#
# # 定义超参数
# sequence_length = 3
# batch_size = 2
#
# # 创建时间序列数据集
# dataset = TimeSeriesOperator.create_time_series_dataset(data, sequence_length, batch_size, shuffle=True)
#
# # 打印数据集
# for batch in dataset:
#     print("Input (X):", batch[0].numpy())
#     print("Target (y):", batch[1].numpy())
