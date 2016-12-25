from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from keras.models import Sequential
import tensorflow as tf
import numpy as np
from keras.layers import Dense, Activation

xy_train = np.genfromtxt('mnist_tr.csv', delimiter=',')
ytrain = xy_train[:,0]
xtrain = xy_train[:,1:]
xy_test = np.genfromtxt('mnist_test.csv', delimiter=',')
ytest = xy_test[:,0]
xtest = xy_test[:,1:]
ytrf = np.zeros((20000,10))
ytef = np.zeros((10000,10))
for i in range(0,20000):
    ytrf[i,ytrain[i]]=1
for i in range(0,10000):
    ytef[i,ytest[i]]=1

model = Sequential()
model.add(Dense(output_dim=50, input_dim=784))
model.add(Activation("sigmoid"))
model.add(Dense(output_dim=50, input_dim=50))
model.add(Activation("tanh"))
model.add(Dense(output_dim=10, input_dim=50))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(xtrain, ytrf, nb_epoch=10, batch_size=32)
loss_and_metrics = model.evaluate(xtest, ytef)

print ("Accuracy of the model is %f percent" % (100*loss_and_metrics[1]))
