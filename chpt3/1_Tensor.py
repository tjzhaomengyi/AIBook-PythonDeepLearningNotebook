# -*- coding: utf-8 -*-
__author__ = 'Mike'
import tensorflow as tf

# 启用 Eager Execution
tf.enable_eager_execution()

x = tf.ones(shape=(2, 1)) #等于np.ones
print(x)

x = tf.zeros(shape=(2, 1)) #等于np.zeros
print(x)

#随机张量
x = tf.random.normal(shape=(3,1), mean=0., stddev=1.)
print(x)

#Variable可以使用assign进行赋值
v = tf.Variable(initial_value=tf.random.normal(shape=(3, 1)))
print(v)
v.assign(tf.ones(shape=(3, 1)))
print(v)

a = tf.ones(shape=(2, 2))
b = tf.square(a)
c = tf.sqrt(a)
d = b + c
e = tf.matmul(a, b)
e *= d
print(e)

'''graident_tape梯度带'''
#tensorflow比numpy牛逼的地方在于对输入的数据可以计算例如梯度等一些数据
input_var = tf.Variable(initial_value=3.)
input_const = tf.constant(3.)
with tf.GradientTape() as tape:
    tape.watch(input_const)
    result = tf.square(input_var)
gradient = tape.gradient(result, input_var)
print(gradient)

#利用tape计算二阶梯度
time = tf.Variable(1.)
with tf.GradientTape() as outer_tape: #帮助记录二阶段梯度信息
    with tf.GradientTape() as inner_tape: #帮助记录一阶梯度信息
        position = 4.9 * time ** 2
    speed = inner_tape.gradient(position, time) #一阶导数
    print(speed)
acceleration = outer_tape.gradient(speed, time) #二阶导数
print(acceleration)



