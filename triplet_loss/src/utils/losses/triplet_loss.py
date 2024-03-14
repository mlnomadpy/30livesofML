import tensorflow as tf

import tensorflow as tf

def loss_function(y_pred, margin=1.0):
    """
    Computes the triplet loss between anchor, positive, and negative embeddings.
    
    Parameters:
        anchor: Tensor, embeddings for anchor samples.
        positive: Tensor, embeddings for positive samples.
        negative: Tensor, embeddings for negative samples.
        margin: Float, margin value for triplet loss.
        
    Returns:
        Tensor, scalar triplet loss.
    """
    anchor, positive, negative = tf.split(y_pred, 3, axis=0)

    # Compute squared Euclidean distances
    ap_distance = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    an_distance = tf.reduce_sum(tf.square(anchor - negative), axis=-1)

    # Compute loss
    loss = tf.maximum(0.0, ap_distance - an_distance + margin)
    
    # Compute mean over batch
    loss = tf.reduce_mean(loss)
    
    return loss


