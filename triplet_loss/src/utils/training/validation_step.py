import tensorflow as tf

@tf.function
def val_step(inputs, model, loss_fn):
    anchor_images, positive_images, negative_images = inputs

    # Compute embeddings for anchor, positive, and negative images
    anchor_embeddings = model(anchor_images, training=False)
    positive_embeddings = model(positive_images, training=False)
    negative_embeddings = model(negative_images, training=False)

    # Calculate triplet loss
    total_v_loss = loss_fn([anchor_embeddings, positive_embeddings, negative_embeddings])

    return total_v_loss
