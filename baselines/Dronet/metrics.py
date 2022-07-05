import tensorflow as tf
import numpy as np

def gate_center_mae_error(y_true, y_pred):
    """Mean average error for gate center prediction.
    
    Args:
        y_true ([tf.float32]): Ground-truth tensor
        y_pred ([tf.float32]): Network prediction.
    
    Returns:
        [tf.float]: Calculated error
    """
    
    p_pred = y_pred[:,:,:,0]
    cx_pred = y_pred[:,:,:,1]
    cy_pred = y_pred[:,:,:,2]

    p_true = y_true[:,:,:,0]
    cx_true = y_true[:,:,:,1]
    cy_true = y_true[:,:,:,2]

    # xy_loss = tf.math.multiply(
    #                         tf.cast(tf.math.logical_and(tf.cast(p_true, tf.bool), tf.math.greater(p_pred,1.)), tf.float32), 
    #                         (tf.math.abs(cx_pred - cx_true)+tf.math.abs(cy_pred - cy_true)))
    xy_loss = tf.math.multiply(tf.cast(p_true, tf.float32), (tf.math.abs(cx_pred - cx_true)+tf.math.abs(cy_pred - cy_true)))    #new IROS 2021

    xy_loss = tf.keras.backend.mean(xy_loss, axis=-1)
    total_loss = xy_loss 
    return total_loss


def distance_mae_error(y_true, y_pred):
    """Mean average error for distance prediction.
    
    Args:
        y_true ([tf.float32]): Ground-truth tensor
        y_pred ([tf.float32]): Network prediction.
    
    Returns:
        [tf.float]: Calculated error
    """
    
    p_pred = y_pred[:,:,:,0]
    d_pred = y_pred[:,:,:,3]

    p_true = y_true[:,:,:,0]
    d_true = y_true[:,:,:,3]

    # d_loss = tf.math.multiply(
    #                         tf.cast(tf.math.logical_and(tf.cast(p_true, tf.bool), tf.math.greater(p_pred,1.)), tf.float32), 
    #                         (tf.math.abs(d_pred - d_true)))
    d_loss = tf.math.multiply(tf.cast(p_true, tf.float32), (tf.math.abs(d_pred - d_true)))    #new IROS 2021

    d_loss = tf.keras.backend.mean(d_loss, axis=-1)
    total_loss = d_loss 
    return total_loss    


def orientation_mae_error(y_true, y_pred):
    """Mean average error for orientation prediction.
    
    Args:
        y_true ([tf.float32]): Ground-truth tensor
        y_pred ([tf.float32]): Network prediction.
    
    Returns:
        [tf.float]: Calculated error
    """
    
    p_pred = y_pred[:,:,:,0]
    o_pred = y_pred[:,:,:,4]

    p_true = y_true[:,:,:,0]
    o_true = y_true[:,:,:,4]

    # o_loss = tf.math.multiply(
    #                         tf.cast(tf.math.logical_and(tf.cast(p_true, tf.bool), tf.math.greater(p_pred,1.)), tf.float32), 
    #                         (tf.math.abs(o_pred - o_true)))
    o_loss = tf.math.multiply(tf.cast(p_true, tf.float32), (tf.math.abs(o_pred - o_true)))    #new IROS 2021

    o_loss = tf.keras.backend.mean(o_loss, axis=-1)
    total_loss = o_loss 
    return total_loss     


def gx_mae_error(y_true, y_pred):
    """Mean average error for distance prediction.
    
    Args:
        y_true ([tf.float32]): Ground-truth tensor
        y_pred ([tf.float32]): Network prediction.
    
    Returns:
        [tf.float]: Calculated error
    """
    
    p_pred = y_pred[:,:,:,0]
    d_pred = y_pred[:,:,:,3]

    p_true = y_true[:,:,:,0]
    d_true = y_true[:,:,:,3]

    d_loss = tf.math.multiply(
                            tf.cast(tf.math.logical_and(tf.cast(p_true, tf.bool), tf.math.greater(p_pred,0.5)), tf.float32), 
                            (tf.math.abs(d_pred - d_true)))

    d_loss = tf.keras.backend.mean(d_loss, axis=-1)
    total_loss = d_loss 
    return total_loss    

def gy_mae_error(y_true, y_pred):
    """Mean average error for distance prediction.
    
    Args:
        y_true ([tf.float32]): Ground-truth tensor
        y_pred ([tf.float32]): Network prediction.
    
    Returns:
        [tf.float]: Calculated error
    """
    
    p_pred = y_pred[:,:,:,0]
    d_pred = y_pred[:,:,:,4]

    p_true = y_true[:,:,:,0]
    d_true = y_true[:,:,:,4]

    d_loss = tf.math.multiply(
                            tf.cast(tf.math.logical_and(tf.cast(p_true, tf.bool), tf.math.greater(p_pred,0.5)), tf.float32), 
                            (tf.math.abs(d_pred - d_true)))

    d_loss = tf.keras.backend.mean(d_loss, axis=-1)
    total_loss = d_loss 
    return total_loss    

def gz_mae_error(y_true, y_pred):
    """Mean average error for distance prediction.
    
    Args:
        y_true ([tf.float32]): Ground-truth tensor
        y_pred ([tf.float32]): Network prediction.
    
    Returns:
        [tf.float]: Calculated error
    """
    
    p_pred = y_pred[:,:,:,0]
    d_pred = y_pred[:,:,:,5]

    p_true = y_true[:,:,:,0]
    d_true = y_true[:,:,:,5]

    d_loss = tf.math.multiply(
                            tf.cast(tf.math.logical_and(tf.cast(p_true, tf.bool), tf.math.greater(p_pred,0.5)), tf.float32), 
                            (tf.math.abs(d_pred - d_true)))

    d_loss = tf.keras.backend.mean(d_loss, axis=-1)
    total_loss = d_loss 
    return total_loss    

def o_mae_error(y_true, y_pred):
    """Mean average error for distance prediction.
    
    Args:
        y_true ([tf.float32]): Ground-truth tensor
        y_pred ([tf.float32]): Network prediction.
    
    Returns:
        [tf.float]: Calculated error
    """
    
    p_pred = y_pred[:,:,:,0]
    d_pred = y_pred[:,:,:,6]

    p_true = y_true[:,:,:,0]
    d_true = y_true[:,:,:,6]

    d_loss = tf.math.multiply(
                            tf.cast(tf.math.logical_and(tf.cast(p_true, tf.bool), tf.math.greater(p_pred,0.5)), tf.float32), 
                            (tf.math.abs(d_pred - d_true)))

    d_loss = tf.keras.backend.mean(d_loss, axis=-1)
    total_loss = d_loss 
    return total_loss    