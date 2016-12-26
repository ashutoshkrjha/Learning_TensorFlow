import keras
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation,Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D

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

I_train = np.zeros(shape=(20000,1,28,28))
I_test = np.zeros(shape=(10000,1,28,28))

for i in range(0,20000):
    I_train[i,0,:,:]=np.reshape(xtrain[i,:],(28,28))

for i in range(0,10000):
    I_test[i,0,:,:]=np.reshape(xtest[i,:],(28,28))

#Params for the CNN
nb_filters = 32
nb_row = 5
nb_col = 5

model = Sequential()

model.add(Convolution2D(nb_filters, 3, 3,border_mode='valid', input_shape=(1,28,28),dim_ordering='th'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, 3, 3,dim_ordering='th'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),dim_ordering='th'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit(I_train, ytrf, nb_epoch=12, batch_size=128)
loss_and_metrics = model.evaluate(I_test, ytef)

print ("Accuracy of the model is %f percent" % (100*loss_and_metrics[1]))
