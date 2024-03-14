#!/usr/bin/env python3

"""
Author: Taha Bouhsine
Email: contact@tahabouhsine.com
Created on: March 03st 2024
Last Modified: March 03st 2024
Description:
This Python script (train.py) is designed for training an image classification model. 
It includes functionality for loading data, setting up a model, and training and evaluating it.
"""

import argparse
import tensorflow as tf
from wandb.keras import WandbCallback
from tensorflow.keras.optimizers import AdamW

import os
import numpy as np
import wandb

import matplotlib.pyplot as plt
import seaborn as sns


from utils.data_loaders.load_data import load_data

from utils.models.build_model import build_model

from utils.miscs.set_gpu import setup_gpus
from utils.miscs.set_seed import set_seed
from utils.training.train_step import train_step
from utils.training.validation_step import val_step

from utils.losses.contrastive_loss import loss_function as loss_fn

def train(config):

    model_name = 'BasicCNN'


    num_classes = config.num_classes
    class_names = [f'bin_{i}' for i in range(num_classes)]
    config.class_names = class_names

    # Log confusion matrix and class-wise metrics to wandb
    class_names = [f'bin_{i}' for i in range(config.num_classes)]
    config.class_labels = class_names


    try:
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = tf.distribute.get_strategy()
        print("Number of accelerators: ", strategy.num_replicas_in_sync)
    except ValueError as e:
        print(f"Error setting up GPU strategy: {e}")
        strategy = tf.distribute.get_strategy()

    wandb.login()

    run = wandb.init(
        project=config.project,
        entity=config.entity,
        name=config.model_name
    )

    wandb.config.update(vars(config))

    print('Loading the Dataset...')
    # Load data
    train_data, val_data = load_data(config)

    print('Classes: ' + str(class_names))



    # Build model
    print('Building Model:')
    with strategy.scope():
        model = build_model(config)
        model.summary()

        optimizer = AdamW()

    wandb_callback = WandbCallback(save_model=False)

    wandb_callback.set_model(model)
    
    best_val_loss = 10000
    # Train for a few epochs
    print("Training Start")
    for epoch in range(config.epochs):
        print(f'Epoch {epoch}:')
        # Reset the metrics at start of each epoch
        # Before each epoch, we call the 'on_epoch_begin' method of our callback
        wandb_callback.on_epoch_begin(epoch)
        if epoch == config.trainable_epochs:
            model.trainable = True

        print('Training...')
        # Training loop
        for steps, (inputs, labels) in enumerate(train_data):
            total_loss = train_step(inputs, labels, model, loss_fn, optimizer)

            # Break the loop once we reach the number of steps per epoch
            if steps >= len(train_data) // config.batch_size:
                break

        train_data.reset()  # Reset the generator after each epoch
        print(
              f'Train Loss: {total_loss}, '
              )

        print('Validation...')
        # Validation loop
        for steps, (inputs, labels) in enumerate(val_data):
            total_v_loss = val_step(inputs, labels, model, loss_fn)

            if steps == len(val_data) // config.batch_size:
                break

        val_data.reset()  # Reset the generator after each epoch
        print(
              f'Val Loss: {total_v_loss}, '
              )


        # Log metrics with Wandb
        logs = {
            'epoch': epoch,
            'train_loss': total_loss,
            'val_loss': total_v_loss,
        }

        wandb_callback.on_epoch_end(epoch, logs=logs)

        current_val_loss = total_v_loss
        if current_val_loss < best_val_loss:
            model.save(os.path.join(wandb.run.dir, f'{model_name}_best_val_loss_model.keras'))
            best_val_loss = current_val_loss

    print('Saving Last Model')
    # Save the trained model
    model.save(os.path.join(wandb.run.dir, f"last_{model_name}_model.keras"))

    print("Training completed successfully!")


if __name__ == '__main__':
    # Parsing command-line arguments
    parser = argparse.ArgumentParser(description='Train a CNN for image classification.')
    parser.add_argument('--img_width', type=int, default=64, help='Image width.')
    parser.add_argument('--img_height', type=int, default=64, help='Image height.')

    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')

    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--test_dataset_path', type=str, required=True, help='Path to the test dataset directory.')
    parser.add_argument('--num_img_lim', type=int, required=True, help='The number of images per class')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of Classes')
    parser.add_argument('--embedding_dim', type=int, required=True, help='Embedding Dimension')
    parser.add_argument('--val_split', type=float, required=True, help='Validation Split')


    parser.add_argument('--seed', type=int, default=64, help='Random Seed.')
    parser.add_argument('--gpu', type=str, default='2,3', help='GPUs.')
    parser.add_argument('--trainable_epochs', type=int, required=True, help='The number of epochs before the backbone become trainable')
    parser.add_argument('--model_name', type=str, required=True, help='Model Name.')
    parser.add_argument('--project', type=str, required=True, help='Wandb Project.')
    parser.add_argument('--entity', type=str, required=True, help='Wandb Entity.')

    args = parser.parse_args()



    # Setup GPUs
    setup_gpus(args.gpu)
    set_seed(args.seed)

    train(args)