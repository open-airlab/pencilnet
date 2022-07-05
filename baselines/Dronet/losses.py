import tensorflow as tf
import numpy as np


# Compute loss
# TODO Implement loss function

def mean_squarred_error(y_true, y_pred):
    mse =  tf.keras.losses.MeanSquaredError()
    return mse(y_true, y_pred)

def my_loss(y_true, y_pred):
    p_pred = y_pred[:,:,:,0]
    cx_pred = y_pred[:,:,:,1]
    cy_pred = y_pred[:,:,:,2]
    #w_pred = y_pred[:,:,:,3]
    #h_pred = y_pred[:,:,:,4]
    d_pred = y_pred[:,:,:,3]
    o_pred = y_pred[:,:,:,4]


    p_true = y_true[:,:,:,0]
    cx_true = y_true[:,:,:,1]
    cy_true = y_true[:,:,:,2]
    #w_true = y_true[:,:,:,3]
    #h_true = y_true[:,:,:,4]
    d_true = y_true[:,:,:,3]
    o_true = y_true[:,:,:,4]

    xy_loss = tf.math.multiply(p_true, tf.math.square(cx_pred - cx_true)+tf.math.square(cy_pred - cy_true))
    #wh_loss = tf.math.multiply(p_true, tf.math.square(tf.math.square(w_pred) - tf.math.square(w_true))+tf.math.square(tf.math.square(h_pred) - tf.math.square(h_true)))

    #d_loss = tf.math.multiply(p_true, tf.math.square(tf.math.sign(d_pred)*tf.math.square(d_pred) - tf.math.sign(d_true)*tf.math.square(d_true))+tf.math.square(tf.math.sign(d_pred)*tf.math.square(d_pred) - tf.math.sign(d_true)*tf.math.square(d_true)))
    #o_loss = tf.math.multiply(p_true, tf.math.square(tf.math.sign(o_pred)*tf.math.square(o_pred) - tf.math.sign(o_true)*tf.math.square(o_true))+tf.math.square(tf.math.sign(o_pred)*tf.math.square(o_pred) - tf.math.sign(o_true)*tf.math.square(o_true)))

    d_loss = tf.math.multiply(p_true, tf.math.square(tf.math.square(d_pred) - tf.math.square(d_true)))
    o_loss = tf.math.multiply(p_true, tf.math.square(tf.math.square(o_pred) - tf.math.square(o_true)))


    p_loss = tf.math.multiply(p_true, tf.math.square(p_pred - p_true))
    nop_loss = tf.math.multiply(1-p_true, tf.math.square(p_pred - p_true))

    xy_loss = tf.keras.backend.sum(xy_loss, axis=-1)
    #wh_loss = tf.keras.backend.sum(wh_loss, axis=-1)
    p_loss = tf.keras.backend.sum(p_loss, axis=-1)
    nop_loss = tf.keras.backend.sum(nop_loss, axis=-1)
    d_loss = tf.keras.backend.sum(d_loss, axis=-1)
    o_loss = tf.keras.backend.sum(o_loss, axis=-1)

    total_loss = 5*xy_loss  + 5*p_loss + 0.5*nop_loss+ 5*d_loss + 5*o_loss

    return total_loss

def absolute_value_loss(y_true, y_pred):
    """ The loss function to eliminate effect of negative signs for distance and orientation """
    p_pred = y_pred[:,:,:,0]
    cx_pred = y_pred[:,:,:,1]
    cy_pred = y_pred[:,:,:,2]
    d_pred = y_pred[:,:,:,3]
    o_pred = y_pred[:,:,:,4]

    p_true = y_true[:,:,:,0]
    cx_true = y_true[:,:,:,1]
    cy_true = y_true[:,:,:,2]
    d_true = y_true[:,:,:,3]
    o_true = y_true[:,:,:,4]

    #xy_loss = tf.math.multiply(p_true, tf.math.square(cx_pred - cx_true)+tf.math.square(cy_pred - cy_true))
    xy_loss = tf.math.multiply(
                            tf.cast(tf.math.logical_and(tf.cast(p_true, tf.bool), tf.math.greater(p_pred,0.5)), tf.float32), 
                            (tf.math.abs(cx_pred - cx_true)+tf.math.abs(cy_pred - cy_true)))


    #d_loss = tf.math.multiply(p_true, tf.math.square(tf.math.square(d_pred) - tf.math.square(d_true)))
    d_loss = tf.math.multiply(
                        tf.cast(tf.math.logical_and(tf.cast(p_true, tf.bool), tf.math.greater(p_pred,0.5)), tf.float32), 
                        (tf.math.abs(tf.math.abs(d_pred) - d_true)))

    #o_loss = tf.math.multiply(p_true, tf.math.square(tf.math.square(o_pred) - tf.math.square(o_true)))
    o_loss = tf.math.multiply(
                            tf.cast(tf.math.logical_and(tf.cast(p_true, tf.bool), tf.math.greater(p_pred,0.5)), tf.float32), 
                            (tf.math.abs(tf.math.abs(o_pred) - o_true)))


    p_loss = tf.math.multiply(p_true, tf.math.square(p_pred - p_true))
    nop_loss = tf.math.multiply(1-p_true, tf.math.square(p_pred - p_true))

    xy_loss = tf.keras.backend.sum(xy_loss, axis=-1)
    p_loss = tf.keras.backend.sum(p_loss, axis=-1)
    nop_loss = tf.keras.backend.sum(nop_loss, axis=-1)
    d_loss = tf.keras.backend.sum(d_loss, axis=-1)
    o_loss = tf.keras.backend.sum(o_loss, axis=-1)

    total_loss = 5*xy_loss  + 5*p_loss + 0.5*nop_loss+ 5*d_loss + 5*o_loss

    return total_loss    



def my_loss_2(y_true, y_pred):
    
    p_pred = y_pred[:,:,:,0]
    cx_pred = y_pred[:,:,:,1]
    cy_pred = y_pred[:,:,:,2]
    #w_pred = y_pred[:,:,:,3]
    #h_pred = y_pred[:,:,:,4]
    d_pred = y_pred[:,:,:,3]
    o_pred = y_pred[:,:,:,4]


    p_true = y_true[:,:,:,0]
    cx_true = y_true[:,:,:,1]
    cy_true = y_true[:,:,:,2]
    #w_true = y_true[:,:,:,3]
    #h_true = y_true[:,:,:,4]
    d_true = y_true[:,:,:,3]
    o_true = y_true[:,:,:,4]

    xy_loss = tf.math.multiply(p_true, tf.math.square(cx_pred - cx_true)+tf.math.square(cy_pred - cy_true))
    #wh_loss = tf.math.multiply(p_true, tf.math.square(tf.math.square(w_pred) - tf.math.square(w_true))+tf.math.square(tf.math.square(h_pred) - tf.math.square(h_true)))

    #d_loss = tf.math.multiply(p_true, tf.math.square(tf.math.sign(d_pred)*tf.math.square(d_pred) - tf.math.sign(d_true)*tf.math.square(d_true))+tf.math.square(tf.math.sign(d_pred)*tf.math.square(d_pred) - tf.math.sign(d_true)*tf.math.square(d_true)))
    #o_loss = tf.math.multiply(p_true, tf.math.square(tf.math.sign(o_pred)*tf.math.square(o_pred) - tf.math.sign(o_true)*tf.math.square(o_true))+tf.math.square(tf.math.sign(o_pred)*tf.math.square(o_pred) - tf.math.sign(o_true)*tf.math.square(o_true)))

    d_loss = tf.math.multiply(p_true, tf.math.square((d_pred) - (d_true)))
    o_loss = tf.math.multiply(p_true, tf.math.square((o_pred) - (o_true)))


    p_loss = tf.math.multiply(p_true, tf.math.square(p_pred - p_true))
    nop_loss = tf.math.multiply(1-p_true, tf.math.square(p_pred - p_true))

    xy_loss = tf.keras.backend.sum(xy_loss, axis=-1)
    #wh_loss = tf.keras.backend.sum(wh_loss, axis=-1)
    p_loss = tf.keras.backend.sum(p_loss, axis=-1)
    nop_loss = tf.keras.backend.sum(nop_loss, axis=-1)
    d_loss = tf.keras.backend.sum(d_loss, axis=-1)
    o_loss = tf.keras.backend.sum(o_loss, axis=-1)

    total_loss = 5*xy_loss  + 5*p_loss + 0.5*nop_loss+ 5*d_loss + 5*o_loss

    return total_loss


def my_loss_2_1(y_true, y_pred):
    
    p_pred = y_pred[:,:,:,0]
    cx_pred = y_pred[:,:,:,1]
    cy_pred = y_pred[:,:,:,2]
    #w_pred = y_pred[:,:,:,3]
    #h_pred = y_pred[:,:,:,4]
    d_pred = y_pred[:,:,:,3]
    o_pred = y_pred[:,:,:,4]


    p_true = y_true[:,:,:,0]
    cx_true = y_true[:,:,:,1]
    cy_true = y_true[:,:,:,2]
    #w_true = y_true[:,:,:,3]
    #h_true = y_true[:,:,:,4]
    d_true = y_true[:,:,:,3]
    o_true = y_true[:,:,:,4]

    xy_loss = tf.math.multiply(p_true, tf.math.square(cx_pred - cx_true)+tf.math.square(cy_pred - cy_true))
    #wh_loss = tf.math.multiply(p_true, tf.math.square(tf.math.square(w_pred) - tf.math.square(w_true))+tf.math.square(tf.math.square(h_pred) - tf.math.square(h_true)))

    #d_loss = tf.math.multiply(p_true, tf.math.square(tf.math.sign(d_pred)*tf.math.square(d_pred) - tf.math.sign(d_true)*tf.math.square(d_true))+tf.math.square(tf.math.sign(d_pred)*tf.math.square(d_pred) - tf.math.sign(d_true)*tf.math.square(d_true)))
    #o_loss = tf.math.multiply(p_true, tf.math.square(tf.math.sign(o_pred)*tf.math.square(o_pred) - tf.math.sign(o_true)*tf.math.square(o_true))+tf.math.square(tf.math.sign(o_pred)*tf.math.square(o_pred) - tf.math.sign(o_true)*tf.math.square(o_true)))

    d_loss = tf.math.multiply(p_true, tf.math.square((d_true) - (d_pred)))
    o_loss = tf.math.multiply(p_true, tf.math.square((o_true) - (o_pred)))


    p_loss = tf.math.multiply(p_true, tf.math.square(p_pred - p_true))
    nop_loss = tf.math.multiply(1-p_true, tf.math.square(p_pred - p_true))

    xy_loss = tf.keras.backend.sum(xy_loss, axis=-1)
    #wh_loss = tf.keras.backend.sum(wh_loss, axis=-1)
    p_loss = tf.keras.backend.sum(p_loss, axis=-1)
    nop_loss = tf.keras.backend.sum(nop_loss, axis=-1)
    d_loss = tf.keras.backend.sum(d_loss, axis=-1)
    o_loss = tf.keras.backend.sum(o_loss, axis=-1)

    total_loss = 5*xy_loss  + 5*p_loss + 0.5*nop_loss+ 5*d_loss + 5*o_loss

    return total_loss


def my_loss_2_2(y_true, y_pred):
    
    p_pred = y_pred[:,:,:,0]
    cx_pred = y_pred[:,:,:,1]
    cy_pred = y_pred[:,:,:,2]
    #w_pred = y_pred[:,:,:,3]
    #h_pred = y_pred[:,:,:,4]
    d_pred = y_pred[:,:,:,3]
    o_pred = y_pred[:,:,:,4]


    p_true = y_true[:,:,:,0]
    cx_true = y_true[:,:,:,1]
    cy_true = y_true[:,:,:,2]
    #w_true = y_true[:,:,:,3]
    #h_true = y_true[:,:,:,4]
    d_true = y_true[:,:,:,3]
    o_true = y_true[:,:,:,4]

    xy_loss = tf.math.multiply(p_true, tf.math.square(cx_true - cx_pred)+tf.math.square(cy_true - cy_pred))
    #wh_loss = tf.math.multiply(p_true, tf.math.square(tf.math.square(w_pred) - tf.math.square(w_true))+tf.math.square(tf.math.square(h_pred) - tf.math.square(h_true)))

    #d_loss = tf.math.multiply(p_true, tf.math.square(tf.math.sign(d_pred)*tf.math.square(d_pred) - tf.math.sign(d_true)*tf.math.square(d_true))+tf.math.square(tf.math.sign(d_pred)*tf.math.square(d_pred) - tf.math.sign(d_true)*tf.math.square(d_true)))
    #o_loss = tf.math.multiply(p_true, tf.math.square(tf.math.sign(o_pred)*tf.math.square(o_pred) - tf.math.sign(o_true)*tf.math.square(o_true))+tf.math.square(tf.math.sign(o_pred)*tf.math.square(o_pred) - tf.math.sign(o_true)*tf.math.square(o_true)))

    d_loss = tf.math.multiply(p_true, tf.math.square((d_true) - (d_pred)))
    o_loss = tf.math.multiply(p_true, tf.math.square((o_true) - (o_pred)))


    p_loss = tf.math.multiply(p_true, tf.math.square(p_pred - p_true))
    nop_loss = tf.math.multiply(1-p_true, tf.math.square(p_pred - p_true))

    xy_loss = tf.keras.backend.sum(xy_loss, axis=-1)
    #wh_loss = tf.keras.backend.sum(wh_loss, axis=-1)
    p_loss = tf.keras.backend.sum(p_loss, axis=-1)
    nop_loss = tf.keras.backend.sum(nop_loss, axis=-1)
    d_loss = tf.keras.backend.sum(d_loss, axis=-1)
    o_loss = tf.keras.backend.sum(o_loss, axis=-1)

    total_loss = 5*xy_loss  + 5*p_loss + 0.5*nop_loss+ 5*d_loss + 5*o_loss

    return total_loss





def my_loss_2_2_large(y_true, y_pred):
    
    p_pred = y_pred[:,:,:,0]
    cx_pred = y_pred[:,:,:,1]
    cy_pred = y_pred[:,:,:,2]
    #w_pred = y_pred[:,:,:,3]
    #h_pred = y_pred[:,:,:,4]
    gx_pred = y_pred[:,:,:,3]
    gy_pred = y_pred[:,:,:,4]
    gz_pred = y_pred[:,:,:,5]

    o_pred = y_pred[:,:,:,6]


    p_true = y_true[:,:,:,0]
    cx_true = y_true[:,:,:,1]
    cy_true = y_true[:,:,:,2]
    #w_true = y_true[:,:,:,3]
    #h_true = y_true[:,:,:,4]
    gx_true = y_true[:,:,:,3]
    gy_true = y_true[:,:,:,4]
    gz_true = y_true[:,:,:,5]
    o_true = y_true[:,:,:,6]

    xy_loss = tf.math.multiply(p_true, tf.math.square(cx_true - cx_pred)+tf.math.square(cy_true - cy_pred))
    #wh_loss = tf.math.multiply(p_true, tf.math.square(tf.math.square(w_pred) - tf.math.square(w_true))+tf.math.square(tf.math.square(h_pred) - tf.math.square(h_true)))

    #d_loss = tf.math.multiply(p_true, tf.math.square(tf.math.sign(d_pred)*tf.math.square(d_pred) - tf.math.sign(d_true)*tf.math.square(d_true))+tf.math.square(tf.math.sign(d_pred)*tf.math.square(d_pred) - tf.math.sign(d_true)*tf.math.square(d_true)))
    #o_loss = tf.math.multiply(p_true, tf.math.square(tf.math.sign(o_pred)*tf.math.square(o_pred) - tf.math.sign(o_true)*tf.math.square(o_true))+tf.math.square(tf.math.sign(o_pred)*tf.math.square(o_pred) - tf.math.sign(o_true)*tf.math.square(o_true)))

    #d_loss = tf.math.multiply(p_true, tf.math.square((d_true) - (d_pred)))
    o_loss = tf.math.multiply(p_true, tf.math.square((o_true) - (o_pred)))

    gx_loss = tf.math.multiply(p_true, tf.math.square((gx_true) - (gx_pred)))
    gy_loss = tf.math.multiply(p_true, tf.math.square((gy_true) - (gy_pred)))
    gz_loss = tf.math.multiply(p_true, tf.math.square((gz_true) - (gz_pred)))


    p_loss = tf.math.multiply(p_true, tf.math.square(p_pred - p_true))
    nop_loss = tf.math.multiply(1-p_true, tf.math.square(p_pred - p_true))

    xy_loss = tf.keras.backend.sum(xy_loss, axis=-1)
    #wh_loss = tf.keras.backend.sum(wh_loss, axis=-1)
    p_loss = tf.keras.backend.sum(p_loss, axis=-1)
    nop_loss = tf.keras.backend.sum(nop_loss, axis=-1)
    #d_loss = tf.keras.backend.sum(d_loss, axis=-1)
    o_loss = tf.keras.backend.sum(o_loss, axis=-1)

    gx_loss = tf.keras.backend.sum(gx_loss, axis=-1)
    gy_loss = tf.keras.backend.sum(gy_loss, axis=-1)
    gz_loss = tf.keras.backend.sum(gz_loss, axis=-1)



    total_loss = 5*xy_loss  + 5*p_loss + 0.5*nop_loss+ 5*gx_loss+ 5*gy_loss+ 5*gz_loss + 5*o_loss

    return total_loss
