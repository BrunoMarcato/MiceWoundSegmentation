'''
FCN models file
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
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

    x = Conv2D(filters=1000,
                kernel_size=(1,1),
                strides=(1,1),
                padding='same',
                activation='softmax',
                name='inference_pred')(x)

    return Model(input, x)



def fcn32s(vgg16, filters=21, weight_decay=0):
    '''
    32x upsampled
    
    Args:
        vgg16: VGG16 model
        fcn16: FCN16 model
        weight_decay = L2 regularization factor (float), weight_decay=0 by default
    returns:
        keras model
    '''

    x = Conv2D(filters=filters, 
                kernel_size=(1,1), 
                strides=(1,1), 
                padding='same', 
                activation='linear',
                kernel_regularizer=regularizers.L2(l2=weight_decay),
                name='score-conv7')(vgg16.get_layer('drop-conv7').output)

    x = Conv2DTranspose(filters=filters, kernel_size=(64,64), strides=(32,32),
                          padding='same', use_bias=False, activation='softmax',
                          kernel_initializer=BilinearInitializer(),
                          kernel_regularizer=regularizers.L2(l2=weight_decay),
                          name='FCN32s')(x)

    return Model(vgg16.input, x)



def fcn16s(vgg16, fcn32, filters=21, weight_decay=0):
    '''
    16x upsampled 
    
    Args:
        vgg16: VGG16 custom keras model
        fcn32: FCN32 custom keras model
        weight_decay = L2 regularization factor (float), weight_decay=0 by default
    returns:
        keras model
    '''
    x = Conv2DTranspose(filters=filters, kernel_size=(32,32), strides=(16,16),
                        padding='same', use_bias=False, activation='linear',
                        kernel_initializer=BilinearInitializer(),
                        kernel_regularizer=regularizers.L2(l2=weight_decay),
                        name='score7_upsample')(vgg16.get_layer('score-conv7').output)

    y = Conv2D(filters=filters, 
                kernel_size=(1,1), 
                strides=(1,1),
                padding='same',  
                activation='linear',
                kernel_initializer=initializers.Zeros(), #Net starts with unmodified predictions
                kernel_regularizer=regularizers.l2(l2=weight_decay) 
                )(vgg16.get_layer('Pool4').output)

    m = Add(name='step4')([x,y]) ##fusion

    m = Conv2DTranspose(filters=filters, kernel_size=(64,64), strides=(32,32),
                                     padding='same', use_bias=False, activation='softmax',
                                     kernel_initializer=BilinearInitializer(),
                                     kernel_regularizer=regularizers.L2(l2=weight_decay),
                                     name='FCN16s')(m)

    return Model(fcn32.input, m)

    

def fcn8s(vgg16, fcn16, filters=21, weight_decay=0):
    '''
    8x upsampled
    
    Args:
        vgg16: VGG16 custom keras model
        fcn16: FCN16 custom keras model
        weight_decay = L2 regularization factor (float), weight_decay=0 by default
    returns:
        keras model
    '''

    x = Conv2DTranspose(filters=filters, kernel_size=(4,4), strides=(2,2),
                                        padding='same', use_bias=False, activation='linear',
                                        kernel_initializer=BilinearInitializer(),
                                        kernel_regularizer=regularizers.L2(l2=weight_decay),
                                        name='skip4_upsample')(fcn16.get_layer('step4').output)

    y = Conv2D(filters=filters, 
                kernel_size=(1,1), 
                strides=(1,1), 
                padding='same', 
                activation='linear', 
                kernel_regularizer=regularizers.l2(l2=weight_decay), 
                )(vgg16.get_layer('Pool3').output)

    m = Add(name='step3')([x,y])

    m = Conv2DTranspose(filters=filters, kernel_size=(16,16), strides=(8,8),
                                     padding='same', use_bias=False, activation='softmax',
                                     kernel_initializer=BilinearInitializer(),
                                     kernel_regularizer=regularizers.L2(l2=weight_decay),
                                     name='FCN8s')(m)

    return Model(fcn16.input, m)

class BilinearInitializer(initializers.Initializer):
  '''Initializer for Conv2DTranspose to perform bilinear interpolation on each channel.'''
  def __call__(self, shape, dtype=None, **kwargs):
      kernel_size, _, filters, _ = shape
      arr = np.zeros((kernel_size, kernel_size, filters, filters))
      ## make filter that performs bilinear interpolation through Conv2DTranspose
      upscale_factor = (kernel_size+1)//2
      if kernel_size % 2 == 1:
          center = upscale_factor - 1
      else:
          center = upscale_factor - 0.5
      og = np.ogrid[:kernel_size, :kernel_size]
      kernel = (1-np.abs(og[0]-center)/upscale_factor) * \
              (1-np.abs(og[1]-center)/upscale_factor) # kernel shape is (kernel_size, kernel_size)
      for i in range(filters):
          arr[..., i, i] = kernel
      return tf.convert_to_tensor(arr, dtype=dtype)
