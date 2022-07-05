import tensorflow as tf
import numpy as np

def standart_model(config):
    # Build the model
    input_tensor = tf.keras.layers.Input(shape=config['input_shape'], name='input')

    conv_1 = tf.keras.layers.Conv2D(16, kernel_size=(3,3),
                                padding='same',
                                kernel_initializer='he_normal',
                                kernel_regularizer=
                                tf.keras.regularizers.l2(config['l2_weight_decay']),
                                bias_regularizer=
                                tf.keras.regularizers.l2(config['l2_weight_decay']),
                                name='conv1')(input_tensor)

    bnorm_1 = tf.keras.layers.BatchNormalization(axis=3,
                                            name='bn1',
                                            momentum=config['batch_norm_decay'],
                                            epsilon=config['batch_norm_epsilon'])(conv_1)

    act_1 = tf.keras.layers.Activation('relu')(bnorm_1)

    pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(act_1)

    conv_2 = tf.keras.layers.Conv2D(32, kernel_size=(3,3),
                                padding='same',
                                kernel_initializer='he_normal',
                                kernel_regularizer=
                                tf.keras.regularizers.l2(config['l2_weight_decay']),
                                bias_regularizer=
                                tf.keras.regularizers.l2(config['l2_weight_decay']),
                                name='conv2')(pool_1)

    bnorm_2 = tf.keras.layers.BatchNormalization(axis=3,
                                            name='bn2',
                                            momentum=config['batch_norm_decay'],
                                            epsilon=config['batch_norm_epsilon'])(conv_2)

    act_2 = tf.keras.layers.Activation('relu')(bnorm_2)

    pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(act_2)

    conv_3 = tf.keras.layers.Conv2D(16, kernel_size=(3,3),
                                padding='same',
                                kernel_initializer='he_normal',
                                kernel_regularizer=
                                tf.keras.regularizers.l2(config['l2_weight_decay']),
                                bias_regularizer=
                                tf.keras.regularizers.l2(config['l2_weight_decay']),
                                name='conv3')(pool_2)

    bnorm_3 = tf.keras.layers.BatchNormalization(axis=3,
                                            name='bn3',
                                            momentum=config['batch_norm_decay'],
                                            epsilon=config['batch_norm_epsilon'])(conv_3)

    act_3 = tf.keras.layers.Activation('relu')(bnorm_3)

    pool_3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(act_3)


    conv_4 = tf.keras.layers.Conv2D(16, kernel_size=(3,3),
                                padding='same',
                                kernel_initializer='he_normal',
                                kernel_regularizer=
                                tf.keras.regularizers.l2(config['l2_weight_decay']),
                                bias_regularizer=
                                tf.keras.regularizers.l2(config['l2_weight_decay']),
                                name='conv4')(pool_3)

    bnorm_4 = tf.keras.layers.BatchNormalization(axis=3,
                                            name='bn4',
                                            momentum=config['batch_norm_decay'],
                                            epsilon=config['batch_norm_epsilon'])(conv_4)

    act_4 = tf.keras.layers.Activation('relu')(bnorm_4)

    pool_4 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(act_4)


    conv_5 = tf.keras.layers.Conv2D(16, kernel_size=(3,3),
                                padding='same',
                                kernel_initializer='he_normal',
                                kernel_regularizer=
                                tf.keras.regularizers.l2(config['l2_weight_decay']),
                                bias_regularizer=
                                tf.keras.regularizers.l2(config['l2_weight_decay']),
                                name='conv5')(pool_4)

    bnorm_5 = tf.keras.layers.BatchNormalization(axis=3,
                                            name='bn5',
                                            momentum=config['batch_norm_decay'],
                                            epsilon=config['batch_norm_epsilon'])(conv_5)

    act_5 = tf.keras.layers.Activation('relu')(bnorm_5)

    pool_5 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(act_5)

    conv_6 = tf.keras.layers.Conv2D(16, kernel_size=(3,3),
                                padding='same',
                                kernel_initializer='he_normal',
                                kernel_regularizer=
                                tf.keras.regularizers.l2(config['l2_weight_decay']),
                                bias_regularizer=
                                tf.keras.regularizers.l2(config['l2_weight_decay']),
                                name='conv6')(pool_5)


    bnorm_6 = tf.keras.layers.BatchNormalization(axis=3,
                                            name='bn6',
                                            momentum=config['batch_norm_decay'],
                                            epsilon=config['batch_norm_epsilon'])(conv_6)

    act_6 = tf.keras.layers.Activation('relu')(bnorm_6)

    #pool_6 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(act_6)
    flatten_6 = tf.keras.layers.Flatten()(act_6)
    #dense_7 = tf.keras.layers.Dense(1024, activation='relu')(flatten_6)
    #drop_7 = tf.keras.layers.Dropout(0.25)(dense_7)



    dense_8 = tf.keras.layers.Dense(np.prod(config['output_shape']), activation='linear')(flatten_6)
    output = tf.keras.layers.Reshape(config['output_shape'])(dense_8)

    model = tf.keras.models.Model([input_tensor], output, name='model')
    return model


def dronet(config):
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


