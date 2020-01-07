# *_*coding:utf-8 *_*
#
import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import DataFrame
from sklearn.utils import shuffle

BATCH_SIZE = 50

iter = 1026 // BATCH_SIZE

print(iter)

names = [
    'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7',
    'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14',
    'f15', 'f16', 'f17', 'c0']

dataset_train: DataFrame = pd.read_csv(filepath_or_buffer='data/traindata.csv', names=names)
dataset_test: DataFrame = pd.read_csv(filepath_or_buffer='data/testdata.csv', names=names)

dataset_train = np.array(dataset_train)
dataset_test = np.array(dataset_test)

dataset_train = shuffle(dataset_train)
dataset_test = shuffle(dataset_test)

x_train, y_train = dataset_train[:, :18], dataset_train[:, 18]
x_test, y_test = dataset_test[:, :18], dataset_test[:, 18]

y_train = tf.one_hot(y_train, 3)
y_test = tf.one_hot(y_test, 3)

print(y_train.shape, y_test.shape)

x_input = tf.placeholder(tf.float32, shape=(None, 18))  # 利用占位符placeholder，输入float32的 一行两列的张量表示 输入n组特征
y_input = tf.placeholder(tf.float32, shape=(None, 3))  # 利用占位符placeholder，输入float32的 一行两列的张量表示 输入n组标签

layer1 = tf.layers.dense(inputs=x_input, units=10, activation=tf.nn.tanh)
layer2 = tf.layers.dense(inputs=layer1, units=14, activation=tf.nn.relu)

logits = tf.layers.dense(inputs=layer2, units=3, activation=tf.nn.softmax)

# 3.定义损失函数及反向出传播方法
print(logits.shape)
loss = tf.reduce_mean(tf.square(logits - y_input))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 梯度下降优化器
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_input, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    # 训练模型
    STEPS = 5000  # 训练5000轮
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % (1026 - BATCH_SIZE)
        end = start + BATCH_SIZE
        sess.run(train_step,
                 feed_dict={x_input: x_train[start:end],
                            y_input: y_train[start:end].eval()})  # 从初始数据集和标签中取出对应的数据，喂入神经网路
        if i % 50 == 0:  # 每500轮打印一次loss值
            total_loss, accuracy_ = sess.run([loss, accuracy], feed_dict={x_input: x_test, y_input: y_test.eval()})
            print('经过{}轮训练, loss所有取值为：{}, accuracy= {}'.format(i, total_loss, accuracy_))
