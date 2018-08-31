from keras.models import Sequential
from keras.models import Model
from keras.layers import Input,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,concatenate,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D
from keras.regularizers import l2
import numpy as np
seed = 7
np.random.seed(seed)

###################################################
############# A version of GoogLenet ##############
###################################################

def Conv2d_All(x, nb_filter,kernel_size, padding='same',strides=(1,1),name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
        
    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
    x = BatchNormalization(axis=3,name=bn_name)(x)
    return x

def Inception(x,nb_filter):
    b1x1 = Conv2d_All(x,nb_filter,(1,1), padding='same',strides=(1,1),name=None)
    b3x3 = Conv2d_All(x,nb_filter,(1,1), padding='same',strides=(1,1),name=None)
    b3x3 = Conv2d_All(b3x3,nb_filter,(3,3), padding='same',strides=(1,1),name=None)
    b5x5 = Conv2d_All(x,nb_filter,(1,1), padding='same',strides=(1,1),name=None)
    b5x5 = Conv2d_All(b5x5,nb_filter,(1,1), padding='same',strides=(1,1),name=None)
    bpool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
    bpool = Conv2d_All(bpool,nb_filter,(1,1),padding='same',strides=(1,1),name=None)
    x = concatenate([b1x1,b3x3,b5x5,bpool],axis=3)
    return x

def CVN(num_classes):
    input1 = Input(shape=(1, 100,80), dtype='float32', name='input1')
    input2 = Input(shape=(1, 100,80), dtype='float32', name='input2')

    x1 = Conv2d_All(input1,64,(7,7),strides=(2,2),padding='same')
    x1 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x1)
    x1 = Conv2d_All(x1,192,(3,3),strides=(1,1),padding='same')
    x1 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x1)
    x1 = Inception(x1,64)
    x1 = Inception(x1,120)
    x1 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x1)
    x1 = Inception(x1,128)
    x1 = AveragePooling2D(pool_size=(7,7),strides=(7,7),padding='same')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(512,activation='relu', W_regularizer=l2(0.1))(x1)

    x2 = Conv2d_All(input2,64,(7,7),strides=(2,2),padding='same')
    x2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x2)
    x2 = Conv2d_All(x2,192,(3,3),strides=(1,1),padding='same')
    x2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x2)
    x2 = Inception(x2,64)
    x2 = Inception(x2,120)
    x2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x2)
    x2 = Inception(x2,128)
    x2 = AveragePooling2D(pool_size=(7,7),strides=(7,7),padding='same')(x2)
    x2 = Flatten()(x2)
    x2 = Dense(512,activation='relu', W_regularizer=l2(0.1))(x2)

    x = concatenate([x1, x2])
    x   = Dense(1024, activation='relu')(x)
    out = Dense(num_classes, activation='softmax', name='out')(x)
    model = Model(inputs=[input1, input2], outputs=[out])

    return model
