import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D, Dropout, Input, Multiply, LayerNormalization, MultiHeadAttention, Concatenate
from tensorflow.keras.models import Model
from utils.models.architectures.densenet import  DenseNet121 as CNN


def build_model(config):
    num_channels = 3

    inputs = Input(shape=(config.img_height, config.img_width, num_channels))
    
    # Define base models for each modality
    base_model_rgb = CNN(input_shape=(config.img_height, config.img_width, num_channels))

    base_model_rgb.trainable = config.trainable_epochs == 0
    x_rgb = base_model_rgb(inputs)
    x_rgb = GlobalAveragePooling2D()(x_rgb)
    x_rgb = LayerNormalization()(x_rgb)
    x_rgb = Dense(config.embedding_dim, activation= tf.keras.activations.gelu)(x_rgb)
    predictions = Dense(config.embedding_dim, activation='tanh')(x_rgb)


    model = Model(inputs=inputs, outputs=predictions)
    return model
