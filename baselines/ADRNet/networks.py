import tensorflow as tf
import numpy as np

def ADRNet(config):
    # Build ADRNet.


    input_tensor = tf.keras.layers.Input(shape=config['input_shape'], name='input')


    conv_1 = tf.keras.layers.Conv2D(96, kernel_size=(11,11), strides= 4,
                    padding= 'valid', activation= 'relu',
                    kernel_initializer= 'he_normal')(input_tensor)


    pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides= (2,2),
                            padding= 'valid', data_format= None)(conv_1)



    conv_2 = tf.keras.layers.Conv2D(256, kernel_size=(5,5), strides= 1,
                    padding= 'same', activation= 'relu',
                    kernel_initializer= 'he_normal')(pool_1)

    pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides= (2,2),
                            padding= 'valid', data_format= None)(conv_2)


    conv_3 = tf.keras.layers.Conv2D(384, kernel_size=(3,3), strides= 1,
                    padding= 'same', activation= 'relu',
                    kernel_initializer= 'he_normal')(pool_2)

    conv_4 = tf.keras.layers.Conv2D(256, kernel_size=(3,3), strides= 1,
                    padding= 'same', activation= 'relu',
                    kernel_initializer= 'he_normal')(conv_3)

    pool_4 = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides= (2,2),
                            padding= 'valid', data_format= None)(conv_4)

    flatten_5 = tf.keras.layers.Flatten()(pool_4)

    dense_5 = tf.keras.layers.Dense(256, activation='relu')(flatten_5)
    dense_6 = tf.keras.layers.Dense(128, activation='relu')(dense_5)


    dense_7 = tf.keras.layers.Dense(np.prod(config['output_shape']), activation='linear')(dense_6)
    output = tf.keras.layers.Reshape(config['output_shape'])(dense_7)



    model = tf.keras.models.Model([input_tensor], output, name='model')
    return model

