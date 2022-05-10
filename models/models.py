'''
FCN models file
'''

from configparser import Interpolation
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers, initializers, Input, Model
from tensorflow.keras.layers import MaxPool2D, Conv2D, Conv2DTranspose, Lambda, Dropout, Add

def vgg16(l2=0, dropout=0):
    '''
    VGG16 network
    
    args:
        l2 = L2 regularization factor (float), l2=0 by default
        dropout = dropout rate (float), dropout=0 by default
        classes = number of classes
    return:
        Keras model
    '''
    
    ##Input as keras tensor
    input = Input(shape=(None, None, 3), name='input')

    ##Preprocessing - should only be loaded in the same environment where they were saved
    
    #x = Lambda(tf.keras.applications.vgg16.preprocess_input, name='preprocessing')(input)

    ##Block 1 - 64 filters
    x = Conv2D(filters = 64,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=l2),
                name='Conv1-1')(input)

    x = Conv2D(filters=64,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=l2),
                name='Conv1-2')(x)

    x = MaxPool2D(pool_size=(2,2),
                    strides=(2,2),
                    name='Pool1')(x)

    ##Block 2 - 128 filters
    x = Conv2D(filters=128,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=l2),
                name='Conv2-1')(x)

    x = Conv2D(filters=128,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=l2),
                name='Conv2-2')(x)
    
    x = MaxPool2D(pool_size=(2,2),
                    strides=(2,2),
                    name='Pool2')(x)
    
    ##Block 3 - 256 filters
    x = Conv2D(filters=256,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=l2),
                name='Conv3-1')(x)

    x = Conv2D(filters=256,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=l2),
                name='Conv3-2')(x)

    x = Conv2D(filters=256,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=l2),
                name='Conv3-3')(x)

    x = MaxPool2D(pool_size=(2,2),
                    strides=(2,2),
                    name='Pool3')(x)

    ##Block 4 - 512 filters
    x = Conv2D(filters=512,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=l2),
                name='Conv4-1')(x)

    x = Conv2D(filters=512,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=l2),
                name='Conv4-2')(x)

    x = Conv2D(filters=512,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=l2),
                name='Conv4-3')(x)

    x = MaxPool2D(pool_size=(2,2),
                    strides=(2,2),
                    name='Pool4')(x)

    ##Block 5 - 512 filters
    x = Conv2D(filters=512,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=l2),
                name='Conv5-1')(x)

    x = Conv2D(filters=512,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=l2),
                name='Conv5-2')(x)

    x = Conv2D(filters=512,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=l2),
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
                kernel_regularizer=regularizers.L2(l2=l2), 
                name='conv6')(x)

    x = Dropout(rate=dropout, name='drop-conv6')(x)

    x = Conv2D(filters=4096, 
                kernel_size=(1,1), 
                strides=(1,1), 
                padding='same', 
                activation='relu',
                kernel_regularizer=regularizers.L2(l2=l2), 
                name='conv7')(x)

    x = Dropout(rate=dropout, name='drop-conv7')(x)

    return Model(input, x)



def fcn32(vgg16, l2=0):
    '''
    32x upsampled
    
    Args:
        vgg16: VGG16 model
        fcn16: FCN16 model
        l2 = L2 regularization factor (float), l2=0 by default

    returns:
        keras model
    '''

    x = Conv2D(filters=21, 
                kernel_size=(1,1), 
                strides=(1,1), 
                padding='same', 
                activation='linear',
                kernel_regularizer=regularizers.L2(l2=l2),
                name='score-conv7')(vgg16.get_layer('drop-conv7').output)

    x = Conv2DTranspose(filters=21, 
                        kernel_size=(64,64), 
                        strides=(32, 32), 
                        padding='same', 
                        use_bias=False, 
                        activation='softmax', 
                        #kernel_initializer=BilinearInterpolation(), 
                        kernel_regularizer=regularizers.l2(l2=l2), 
                        name='FCN32s')(x)

    return Model(vgg16.input, x)



def fcn16(vgg16, fcn32, l2=0):
    '''
    16x upsampled 
    
    Args:
        vgg16: VGG16 custom keras model
        fcn32: FCN32 custom keras model
        l2 = L2 regularization factor (float), l2=0 by default

    returns:
        keras model
    '''
    x = Conv2DTranspose(filters=21, 
                        kernel_size=(4,4), 
                        strides=(2,2), 
                        padding='same', 
                        activation='linear', 
                        kernel_regularizer=regularizers.l2(l2=l2),
                        name='upsampled-conv7')(vgg16.get_layer('score-conv7').output)

    y = Conv2D(filters=21, 
                kernel_size=(1,1), 
                strides=(1,1),
                padding='same',  
                activation='linear',
                kernel_initializer=initializers.Zeros(), #Net starts with unmodified predictions
                kernel_regularizer=regularizers.l2(l2=l2) 
                )(vgg16.get_layer('Pool4').output)

    m = Add(name='fusion4')([x,y]) ##fusion

    m  = Conv2DTranspose(filters=21, 
                            kernel_size=(32,32), 
                            strides=(16,16), 
                            padding='same', 
                            activation='softmax',
                            use_bias=False, 
                            #kernel_initializer=BilinearInterpolation(), 
                            kernel_regularizer=regularizers.l2(l2=l2), 
                            name='FCN16s')(m)

    return Model(fcn32.input, m)

    

def fcn8(vgg16, fcn16, l2=0):
    '''
    8x upsampled
    
    Args:
        vgg16: VGG16 custom keras model
        fcn16: FCN16 custom keras model
        l2 = L2 regularization factor (float), l2=0 by default

    returns:
        keras model
    '''

    x = Conv2DTranspose(filters=21, 
                        kernel_size=(4,4), 
                        strides=(2,2), 
                        padding='same',
                        activation='linear', 
                        kernel_regularizer=regularizers.l2(l2=l2), 
                        name='upsampled-fusion')(fcn16.get_layer('fusion4').output)

    y = Conv2D(filters=21, 
                kernel_size=(1,1), 
                strides=(1,1), 
                padding='same', 
                activation='linear', 
                kernel_regularizer=regularizers.l2(l2=l2), 
                )(vgg16.get_layer('Pool3').output)

    m = Add(name='fusion3')([x,y])

    m = Conv2DTranspose(filters=21, 
                        kernel_size=(16,16), 
                        strides=(8,8), 
                        padding='same', 
                        activation='softmax', 
                        #kernel_initializer=BilinearInterpolation(),
                        kernel_regularizer=regularizers.l2(l2=l2), 
                        name='FCN-8s')(m)

    return Model(fcn16.input, m)