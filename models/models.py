'''
FCN models file
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers, initializers, Input, Model
from tensorflow.keras.layers import MaxPool2D, Conv2D, Conv2DTranspose, Lambda, Dropout, Add, UpSampling2D

def vgg16(weight_decay=0, dropout=0.5):
    '''
    VGG16 network
    
    args:
        weight_decay = L2 regularization factor (float), weight_decay=0 by default
        dropout = dropout rate (float), dropout=0.5 by default
        classes = number of classes
    return:
        Keras model
    '''
    
    ##Input as keras tensor
    input = Input(shape=(None, None, 3), name='input')

    ##Block 1 - 64 filters
    x = Conv2D(filters = 64,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=weight_decay),
                name='Conv1-1')(input)

    x = Conv2D(filters=64,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=weight_decay),
                name='Conv1-2')(x)

    x = MaxPool2D(pool_size=(2,2),
                    strides=(2,2),
                    name='Pool1')(x)

    ##Block 2 - 128 filters
    x = Conv2D(filters=128,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=weight_decay),
                name='Conv2-1')(x)

    x = Conv2D(filters=128,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=weight_decay),
                name='Conv2-2')(x)
    
    x = MaxPool2D(pool_size=(2,2),
                    strides=(2,2),
                    name='Pool2')(x)
    
    ##Block 3 - 256 filters
    x = Conv2D(filters=256,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=weight_decay),
                name='Conv3-1')(x)

    x = Conv2D(filters=256,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=weight_decay),
                name='Conv3-2')(x)

    x = Conv2D(filters=256,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=weight_decay),
                name='Conv3-3')(x)

    x = MaxPool2D(pool_size=(2,2),
                    strides=(2,2),
                    name='Pool3')(x)

    ##Block 4 - 512 filters
    x = Conv2D(filters=512,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=weight_decay),
                name='Conv4-1')(x)

    x = Conv2D(filters=512,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=weight_decay),
                name='Conv4-2')(x)

    x = Conv2D(filters=512,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=weight_decay),
                name='Conv4-3')(x)

    x = MaxPool2D(pool_size=(2,2),
                    strides=(2,2),
                    name='Pool4')(x)

    ##Block 5 - 512 filters
    x = Conv2D(filters=512,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=weight_decay),
                name='Conv5-1')(x)

    x = Conv2D(filters=512,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=weight_decay),
                name='Conv5-2')(x)

    x = Conv2D(filters=512,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=weight_decay),
                name='Conv5-3')(x)

    x = MaxPool2D(pool_size=(2,2),
                    strides=(2,2),
                    name='Pool5')(x)

    ## FC --> Convolutionized Fully Connected Layers

    x = Conv2D(filters=4096, 
                kernel_size=(7,7), 
                strides=(1,1), 
                padding='same', 
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=weight_decay), 
                name='conv6')(x)

    x = Dropout(rate=dropout, name='drop-conv6')(x)

    x = Conv2D(filters=4096, 
                kernel_size=(1,1), 
                strides=(1,1), 
                padding='same', 
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=weight_decay), 
                name='conv7')(x)

    x = Dropout(rate=dropout, name='drop-conv7')(x)

    return Model(input, x)



def fcn32s(vgg16, weight_decay=0):
    '''
    32x upsampled
    
    Args:
        vgg16: VGG16 model
        fcn16: FCN16 model
        weight_decay = L2 regularization factor (float), weight_decay=0 by default

    returns:
        keras model
    '''

    x = Conv2D(filters=21, 
                kernel_size=(1,1), 
                strides=(1,1), 
                padding='same', 
                activation='linear',
                kernel_regularizer=regularizers.L2(l2=weight_decay),
                name='score-conv7')(vgg16.get_layer('drop-conv7').output)

    x = UpSampling2D(size=(32,32), interpolation='bilinear', name='upsample-32')(x)

    x = Conv2D(filters=21, 
                kernel_size=(1,1),
                strides=(1,1),
                padding='same',
                activation='linear',
                kernel_regularizer=regularizers.L2(l2=weight_decay),
                name='FCN32s')(x)

    return Model(vgg16.input, x)



def fcn16s(vgg16, fcn32, weight_decay=0):
    '''
    16x upsampled 
    
    Args:
        vgg16: VGG16 custom keras model
        fcn32: FCN32 custom keras model
        weight_decay = L2 regularization factor (float), weight_decay=0 by default

    returns:
        keras model
    '''
    x = UpSampling2D(size=(2,2), interpolation='bilinear')(vgg16.get_layer('drop-conv7').output)

    x = Conv2D(filters=21, 
                kernel_size=(1,1),
                strides=(1,1),
                padding='same',
                activation='linear',
                kernel_regularizer=regularizers.L2(l2=weight_decay),
                name='upsample-conv7')(x)

    y = Conv2D(filters=21, 
                kernel_size=(1,1), 
                strides=(1,1),
                padding='same',  
                activation='linear',
                kernel_initializer=initializers.Zeros(), #Net starts with unmodified predictions
                kernel_regularizer=regularizers.l2(l2=weight_decay) 
                )(vgg16.get_layer('Pool4').output)

    m = Add(name='step4')([x,y]) ##fusion

    m  = UpSampling2D(size=(16,16), interpolation='bilinear', name='FCN16s')(m)

    x = Conv2D(filters=21, 
                kernel_size=(1,1),
                strides=(1,1),
                padding='same',
                activation='linear',
                kernel_regularizer=regularizers.L2(l2=weight_decay),
                name='FCN16s')(m)

    return Model(fcn32.input, m)

    

def fcn8s(vgg16, fcn16, weight_decay=0):
    '''
    8x upsampled
    
    Args:
        vgg16: VGG16 custom keras model
        fcn16: FCN16 custom keras model
        weight_decay = L2 regularization factor (float), weight_decay=0 by default

    returns:
        keras model
    '''

    x = UpSampling2D(size=(2,2), interpolation='bilinear', name='upsampled-step4')(fcn16.get_layer('step4').output)

    x = Conv2D(filters=21, 
                kernel_size=(1,1),
                strides=(1,1),
                padding='same',
                activation='linear',
                kernel_regularizer=regularizers.L2(l2=weight_decay),
                name='upsample-step4')(x)

    y = Conv2D(filters=21, 
                kernel_size=(1,1), 
                strides=(1,1), 
                padding='same', 
                activation='linear', 
                kernel_regularizer=regularizers.l2(l2=weight_decay), 
                )(vgg16.get_layer('Pool3').output)

    m = Add(name='step3')([x,y])

    m = UpSampling2D(size=(8,8), interpolation='bilinear', name='upsampled-step4')(fcn16.get_layer('step4').output)

    m = Conv2D(filters=21, 
                kernel_size=(1,1),
                strides=(1,1),
                padding='same',
                activation='linear',
                kernel_regularizer=regularizers.L2(l2=weight_decay),
                name='upsample-step4')(m)

    return Model(fcn16.input, m)