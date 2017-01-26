from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

xy_train = np.genfromtxt('mnist_tr.csv', delimiter=',')
ytrain = xy_train[:,0]
xtrain = xy_train[:,1:]
xy_test = np.genfromtxt('mnist_test.csv', delimiter=',')
ytest = xy_test[:,0]
xtest = xy_test[:,1:]
x = tf.placeholder(tf.float32, [None, 784])
W1 = tf.Variable(tf.zeros([784, 10]))
b1 = tf.Variable(tf.zeros([10]))
y = tf.matmul(x,W1) + b1
#a_1 = tf.nn.relu(tf.matmul(x, W1) + b1)

#W2 = tf.Variable(tf.zeros([50,10]))
#b2 = tf.Variable(tf.zeros([10]))
#y = tf.matmul(a_1, W2) + b2

y_ = tf.placeholder(tf.float32, [None, 10])

ytrf = np.zeros((20000,10))
ytef = np.zeros((10000,10))

for i in range(0,20000):
    ytrf[i,ytrain[i]]=1

for i in range(0,10000):
    ytef[i,ytest[i]]=1

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(1000):               #Run for 1000 epochs on full data. No batch GD.
    sess.run(train_step, feed_dict={x: xtrain, y_: ytrf})
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: xtest, y_: ytef}))
