# coding: utf-8
import pandas as pd
df=pd.read_csv('train(1).csv')
import numpy as np
y=df[[0]]
df.drop('label',axis=1,inplace=True)
y=np.array(y.values)
y
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D, MaxPooling2D
df.shape
train=np.array(df)
train=train.astype('float32')
train
train[0]
train=train/255
train[0]
train.shape
y.shape
y_train=np_utils.to_categorical(y,10)
y_train
model=Sequential()
import keras
model=Sequential()
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
model=Sequential()
model.add(Dense(512,input_dim=(784,))
)
model.add(Dense(512,input_dim=(784,)))
model.add(Dense(512,input_dim=784))
model.add(Activation('relu'))
model.add(Dropout(.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(.25))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer=rms)
model.compile(loss='categorical_crossentropy',optimizer=adadelta)
model.compile(loss='categorical_crossentropy',optimizer=)
from keras.optimizers import SGD, Adam, RMSprop
from keras.optimizers import SGD, Adadelta, Adagrad
model.compile(loss='categorical_crossentropy',optimizer=adadelta)
model.compile(loss='categorical_crossentropy',optimizer='adadelta')
model.fit(train,y
)
y
label
y_train
y_train.shape
model.fit(train,y_train,show_accuracy=True,verbose=True)
test=pd.read_csv('test.csv')
test.shape
test=np.array(test).astype('float32')
test
test[0]
test=test/255
test[0]
output=model.predict_classes(test)
output
s=pd.read_csv('final.csv')
s
s.Label=output
s.to_csv('final.csv',index=False)
get_ipython().magic(u'save 1-64 hist.py')
