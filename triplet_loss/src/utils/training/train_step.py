import tensorflow as tf

@tf.function
def train_step(inputs, model, loss_fn, optimizer):
    anchor_images, positive_images, negative_images = inputs

    with tf.GradientTape() as tape:
        # Get the embeddings for anchor, positive, and negative images
        anchor_embeddings = model(anchor_images, training=True)
        positive_embeddings = model(positive_images, training=True)
        negative_embeddings = model(negative_images, training=True)

        # Calculate triplet loss
        total_loss = loss_fn([anchor_embeddings, positive_embeddings, negative_embeddings])

    # Compute gradients and update weights
    gradients = tape.gradient(total_loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss
