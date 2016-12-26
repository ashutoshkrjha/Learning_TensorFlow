import keras
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation,Flatten

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
nb_filter = 6
nb_row = 5
nb_col = 5


model = Sequential()
#First convolutional layer
#model.add(keras.layers.convolutional.Convolution2D(nb_filter, nb_row, nb_col, init='glorot_uniform', activation=None, weights=None, border_mode='valid', subsample=(1, 1), dim_ordering='th', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_shape=(1, 28, 28)))
model.add(keras.layers.convolutional.Convolution2D(nb_filter, nb_row, nb_col, border_mode='same', input_shape=(1, 28, 28)))
model.add(Activation('relu'))

#Sub-sampling layer
model.add(keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th'))
#model.add(keras.layers.pooling.MaxPooling2D(pool_size=(2,2)))

#Second convolutional layer
#model.add(keras.layers.convolutional.Convolution2D(16, nb_row, nb_col, init='glorot_uniform', activation=None, weights=None, border_mode='valid', subsample=(1, 1), dim_ordering='default', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True))
model.add(keras.layers.convolutional.Convolution2D(16, nb_row, nb_col, border_mode='same'))
model.add(Activation('relu'))

#Second Sub-sampling layer
model.add(keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th'))
#model.add(keras.layers.pooling.MaxPooling2D(pool_size=(2,2)))

#Third Convolution layer
#model.add(keras.layers.convolutional.Convolution2D(120, nb_row, nb_col, init='glorot_uniform', activation=None, weights=None, border_mode='valid', subsample=(1, 1), dim_ordering='default', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True))
#model.add(keras.layers.convolutional.Convolution2D(120, nb_row, nb_col, border_mode='same'))
#model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(output_dim=128))
model.add(Activation("sigmoid"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(I_train, ytrf, nb_epoch=10, batch_size=128)
loss_and_metrics = model.evaluate(I_test, ytef)

print ("Accuracy of the model is %f percent" % (100*loss_and_metrics[1]))
