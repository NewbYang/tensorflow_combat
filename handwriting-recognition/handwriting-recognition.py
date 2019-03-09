# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 16:45:01 2019

@author: yang
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

print("****************begin download*****************")
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
print("****************finish download*****************")

print(mnist.train.images.shape,mnist.train.labels.shape)
print(mnist.test.images.shape,mnist.test.labels.shape)
print(mnist.validation.images.shape,mnist.test.labels.shape)

sess = tf.InteractiveSession()

"""
#定义算法公式
x = tf.placeholder(tf.float32,[None,784])
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros(([10])))
y = tf.nn.softmax(tf.matmul(x,w)+b)
"""

x = tf.placeholder(tf.float32,[None,784])
w1 = tf.Variable(tf.truncated_normal([784,512],stddev =0.1))
b1 = tf.Variable(tf.zeros([512]))
hidden1 = tf.nn.sigmoid(tf.matmul(x,w1)+b1)
hidden1_drop = tf.nn.dropout(hidden1,0.5)

w2 = tf.Variable(tf.zeros([512,10]))
b2 = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(hidden1_drop,w2)+b2)


#定义loss
y_=tf.placeholder(tf.float32,[None,10])
cross_entory = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))

#定义优化函数
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entory)

#权重初始化
tf.global_variables_initializer().run()

#迭代的对数据进行训练
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accury = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
for i in  range(5000):
    print("time: ",i)
    batch_xs, batch_ys = nist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})
    print(accury.eval({x: mnist.validation.images, y_: mnist.validation.labels}))

print(accury.eval({x: mnist.test.images, y_: mnist.test.labels}))
