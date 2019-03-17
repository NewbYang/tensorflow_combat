# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 21:25:19 2019

@author: yang
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as  tf

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
sess = tf.InteractiveSession()

def weight_variable(shape):
    inital= tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital)

def bias_variable(shape):
    inital= tf.constant(0.1, shape=shape)
    return tf.Variable(inital)

def  conv2d(x,w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

#卷积1
w_conv1 = weight_variable([5, 5, 1,32])
b_conv1 = bias_variable([32])
#激活1
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1)+b_conv1)
#下采样1
h_pool1 = max_pool_2x2(h_conv1)

#卷积2
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
#激活
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2)+b_conv2)
#下采样2
h_pool2 = max_pool_2x2(h_conv2)

#全连接1
w_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat =tf.reshape(h_pool2,[-1,7*7*64])
#激活1
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1)+b_fc1)

#dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#全连接2
w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
h_fc2 = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2)+b_fc2)

#定义交叉熵
cross_entory = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(h_fc2),reduction_indices=[1]))
#定义优化算法
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entory)

correct_prodiction = tf.equal(tf.argmax(h_fc2, 1),  tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prodiction, tf.float32))

#初始化参数
tf.global_variables_initializer().run()

for i in range(3000):
    batch = mnist.train.next_batch(50)
    if i%100==0:
        train_accury = accuracy.eval(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d,trainning accury %g"%(i, train_accury))
    train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob:0.5})

print("test accury %g"%accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))



