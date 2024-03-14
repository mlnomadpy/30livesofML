import tensorflow as tf

@tf.function
def val_step(inputs, labels, model, loss_fn):
    anchor_images, contrastive_images, labels = inputs

    # Compute embeddings for anchor, positive, and negative images
    anchor_embeddings = model(anchor_images, training=False)
    contrastive_embeddings = model(contrastive_images, training=False)

    # Calculate triplet loss
    total_v_loss = loss_fn(anchor_embeddings, contrastive_embeddings, labels)

    return total_v_loss
