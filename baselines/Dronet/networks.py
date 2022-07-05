import tensorflow as tf
import numpy as np

def Dronet(config):
    """ Dronet """
    input_tensor = tf.keras.layers.Input(shape=config['input_shape'], name='input')
    conv_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=2, padding='same', activation='linear')(input_tensor)
    pool_1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(conv_1)  # default pool_size='2', strides=2

    # First residual block
    bnorm_2 = tf.keras.layers.BatchNormalization()(pool_1)
    relu_2  = tf.keras.layers.Activation('relu')(bnorm_2)   
    conv_2  = tf.keras.layers.Conv2D(filters=32, 
                            kernel_size=3, strides=2, padding='same', 
                            activation='linear', kernel_initializer='he_normal', 
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4))(relu_2)

    bnorm_3 = tf.keras.layers.BatchNormalization()(conv_2)
    relu_3  = tf.keras.layers.Activation('relu')(bnorm_3)
    conv_3  = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', 
                            activation='linear', kernel_initializer='he_normal', 
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4))(relu_3)

    conv_4  = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=2, padding='same', activation='linear')(pool_1)
    add_4   = tf.keras.layers.add([conv_4, conv_3])

    # Second residual block
    bnorm_5 = tf.keras.layers.BatchNormalization()(add_4)
    relu_5  = tf.keras.layers.Activation('relu')(bnorm_5)

    conv_5  = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', 
                                activation='linear', kernel_initializer='he_normal', 
                                kernel_regularizer=tf.keras.regularizers.l2(1e-4))(relu_5)

    bnorm_6 = tf.keras.layers.BatchNormalization()(conv_5)
    relu_6  = tf.keras.layers.Activation('relu')(bnorm_6)

    conv_7  = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', 
                            activation='linear', kernel_initializer='he_normal', 
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4))(relu_6)

    conv_8  = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=2, padding='same', activation='linear')(add_4)
    add_8   = tf.keras.layers.add([conv_8, conv_7])

    # Third residual block
    bnorm_9 = tf.keras.layers.BatchNormalization()(add_8)
    relu_9  = tf.keras.layers.Activation('relu')(bnorm_9)

    conv_9  = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='linear', 
                                        kernel_initializer='he_normal', 
                                        kernel_regularizer=tf.keras.regularizers.l2(1e-4))(relu_9)
    
    bnorm_10 = tf.keras.layers.BatchNormalization()(conv_9)

    relu_10  = tf.keras.layers.Activation('relu')(bnorm_10)
    conv_11  = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='linear', 
                                            kernel_initializer='he_normal', 
                                            kernel_regularizer=tf.keras.regularizers.l2(1e-4))(relu_10)

    conv_12   = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=2, padding='same', activation='linear')(add_8)

    add_13   = tf.keras.layers.add([conv_12, conv_11])

    flatten_14 = tf.keras.layers.Flatten()(add_13)

    dense0 = tf.keras.layers.Dense(units=64, activation='relu')(flatten_14)
    dense1 = tf.keras.layers.Dense(units=32, activation='relu')(dense0)

    dense2 = tf.keras.layers.Dense(np.prod(config['output_shape']), activation='linear')(dense1)
    output = tf.keras.layers.Reshape(config['output_shape'])(dense2)

    model = tf.keras.models.Model([input_tensor], output, name='model')
    return model


def Dronet_half(config):
    """ Dronet half cappacity"""
    input_tensor = tf.keras.layers.Input(shape=config['input_shape'], name='input')
    conv_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=2, padding='same', activation='linear')(input_tensor)
    pool_1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(conv_1)  # default pool_size='2', strides=2

    # First residual block
    bnorm_2 = tf.keras.layers.BatchNormalization()(pool_1)
    relu_2  = tf.keras.layers.Activation('relu')(bnorm_2)   
    conv_2  = tf.keras.layers.Conv2D(filters=16, 
                            kernel_size=3, strides=2, padding='same', 
                            activation='linear', kernel_initializer='he_normal', 
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4))(relu_2)

    bnorm_3 = tf.keras.layers.BatchNormalization()(conv_2)
    relu_3  = tf.keras.layers.Activation('relu')(bnorm_3)
    conv_3  = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', 
                            activation='linear', kernel_initializer='he_normal', 
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4))(relu_3)

    conv_4  = tf.keras.layers.Conv2D(filters=16, kernel_size=1, strides=2, padding='same', activation='linear')(pool_1)
    add_4   = tf.keras.layers.add([conv_4, conv_3])

    # Second residual block
    bnorm_5 = tf.keras.layers.BatchNormalization()(add_4)
    relu_5  = tf.keras.layers.Activation('relu')(bnorm_5)

    conv_5  = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', 
                                activation='linear', kernel_initializer='he_normal', 
                                kernel_regularizer=tf.keras.regularizers.l2(1e-4))(relu_5)

    bnorm_6 = tf.keras.layers.BatchNormalization()(conv_5)
    relu_6  = tf.keras.layers.Activation('relu')(bnorm_6)

    conv_7  = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', 
                            activation='linear', kernel_initializer='he_normal', 
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4))(relu_6)

    conv_8  = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=2, padding='same', activation='linear')(add_4)
    add_8   = tf.keras.layers.add([conv_8, conv_7])

    # Third residual block
    bnorm_9 = tf.keras.layers.BatchNormalization()(add_8)
    relu_9  = tf.keras.layers.Activation('relu')(bnorm_9)

    conv_9  = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='linear', 
                                        kernel_initializer='he_normal', 
                                        kernel_regularizer=tf.keras.regularizers.l2(1e-4))(relu_9)
    
    bnorm_10 = tf.keras.layers.BatchNormalization()(conv_9)

    relu_10  = tf.keras.layers.Activation('relu')(bnorm_10)
    conv_11  = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='linear', 
                                            kernel_initializer='he_normal', 
                                            kernel_regularizer=tf.keras.regularizers.l2(1e-4))(relu_10)

    conv_12   = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=2, padding='same', activation='linear')(add_8)

    add_13   = tf.keras.layers.add([conv_12, conv_11])

    flatten_14 = tf.keras.layers.Flatten()(add_13)

    dense0 = tf.keras.layers.Dense(units=32, activation='relu')(flatten_14)
    dense1 = tf.keras.layers.Dense(units=16, activation='relu')(dense0)

    dense2 = tf.keras.layers.Dense(np.prod(config['output_shape']), activation='linear')(dense1)
    output = tf.keras.layers.Reshape(config['output_shape'])(dense2)

    model = tf.keras.models.Model([input_tensor], output, name='model')
    return model