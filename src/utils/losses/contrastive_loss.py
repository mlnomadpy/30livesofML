import tensorflow as tf


def loss_function(anchor, contrastive, labels, margin=1.0):
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
    # Compute squared Euclidean distances
    distance = tf.reduce_sum(tf.square(anchor - contrastive), axis=-1)

    # Compute loss 
    # 1 for same 0 for different 
    loss = labels * tf.square(distance) + (1.0 - labels) * tf.maximum(0.0, margin -  tf.square(distance))
    
    # Compute mean over batch
    loss = tf.reduce_mean(loss)
    
    return loss


