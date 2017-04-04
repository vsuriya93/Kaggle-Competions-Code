# coding: utf-8
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
batch_size = 32
nb_classes = 10
nb_epoch = 200
data_augmentation = True
# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3
import pandas as pd
df=pd.read_csv('train.csv',header=None)
import numpy as np
label=pd.read_csv('trainLabels.csv')
count=0
mapper={}
for x in label.label.unique():
    mapper[x]=count
    count=count+1
    
label.label.replace(mapper,inplace=True)
label=np.array(label.label)
from sklearn.cross_validation import train_test_split
train,test,train_label,test_label=train_test_split(df,label,test_size=.20,random_state=33)
y_train=np_utils.to_categorical(train_label,nb_classes)
y_test=np_utils.to_categorical(test_label,nb_classes)
"""
train=train.reshape(-1,3,32,32)
test=test.reshape(-1,3,32,32)
train=train.astype('float32')
test=test.astype('float32')
train=train/255
test=test/255

model=Sequential()

model.add(Convolution2D(32,3,3,border_mode='valid',input_shape=(3,32,32)))
model.add(Activation('relu'))
model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.25))
model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(.5))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adadelta')
model.fit(train,y_train,nb_epoch=15,show_accuracy=True,verbose=1,validation_data=(test,y_test))
"""
model=Sequential()
model.add(Convolution2D(32,3,3,border_mode='same',input_shape=(3,32,32)))
model.add(Activation('tanh'))
model.add(Convolution2D(32, 3, 3))
model.add(Dense(nb_classes))
#model.add(MaxPooling2D(pool_size=(2 2)))
model.add(Activation('softmax'))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
model.fit(train,y_train,batch_size=32,nb_epoch=200)
ouput=model.predict_classes(test)
