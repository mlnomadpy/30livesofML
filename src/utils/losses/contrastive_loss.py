import tensorflow as tf


def loss_function(anchor, contrastive, labels, margin=1.0):
    """
    Computes the triplet loss between anchor, contrastive (either a postive or a negative embedding) embeddings.
    
    Parameters:
        anchor: Tensor, embeddings for anchor samples.
        contrastive: Tensor embeddings samples.
        labels: Tensor for match or different (1 if the anchor and contrastive are the same class, 0 otherwise) samples.
        margin: Float, margin value for contrastive loss.
        
    Returns:
        Tensor, scalar loss.
    """
    # Compute squared Euclidean distances
    distance = tf.reduce_sum(tf.square(anchor - contrastive), axis=-1)

    # Compute loss 
    # labels are 1 for same 0 for different 
    loss = labels * tf.square(distance) + (1.0 - labels) * tf.maximum(0.0, margin -  tf.square(distance))
    
    # Compute mean over batch
    loss = tf.reduce_mean(loss)
    
    return loss


