from keras.utils import Sequence
import os
import pandas as pd
import random
import tensorflow as tf
import numpy as np

def limit_data(config, n=100000):
    a = []
    
    rgb_dir = config.dataset_path

    print(f"RGB Directory: {rgb_dir}")

    files_list = os.listdir(rgb_dir)
    random.shuffle(files_list)

    for folder_name in files_list:
        rgb_folder = os.path.join(rgb_dir, folder_name)

        if not os.path.isdir(rgb_folder):
            print(f"Missing folder for class '{folder_name}'. Check RGB directories.")
            continue

        rgb_files = os.listdir(rgb_folder)
        for k, rgb_file in enumerate(rgb_files):
            if k >= n:
                break

            rgb_path = os.path.join(rgb_folder, rgb_file)

            a.append((rgb_path, folder_name))

    df = pd.DataFrame(a, columns=['rgb', 'class'])
    print(f"Total image found: {len(df)}")
    return df


class MultiModalDataGenerator(Sequence):
    def __init__(self, df, config, subset):
        self.df = df
        self.batch_size = config.batch_size
        self.target_size = (config.img_height, config.img_width)
        self.subset = subset
        self.config = config

        if subset == 'training':
            self.df = self.df.sample(frac=1-config.val_split, random_state=config.seed)
        elif subset == 'validation':
            self.df = self.df.drop(self.df.sample(frac=1-config.val_split, random_state=config.seed).index)
        
        self.log_image_counts()

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))

    def log_image_counts(self):
        self.num_images = len(self.df)
        counts_per_set = self.df['class'].value_counts()
        print(f"Total image sets in {self.subset} set:")
        print(counts_per_set)

    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]

        anchor_images, contrastive_images, labels = [], [], []
        probabilities = [0.5, 0.5]

        for _, anchor_row in batch_df.iterrows():
            anchor_img = tf.io.read_file(anchor_row['rgb'])
            anchor_img = tf.image.decode_png(anchor_img, channels=3)
            anchor_img = tf.image.resize(anchor_img, self.target_size)
            anchor_images.append(anchor_img)

            is_positive = random.choice([True, False], weights=probabilities, k=1)[0]
            if is_positive:

                # Selecting positive sample from the same class
                positive_df = self.df[self.df['class'] == anchor_row['class']]
                positive_df = positive_df.sample(frac = 1)

                positive_row = positive_df.iloc[np.random.randint(0, len(positive_df))]
                positive_img = tf.io.read_file(positive_row['rgb'])
                positive_img = tf.image.decode_png(positive_img, channels=3)
                positive_img = tf.image.resize(positive_img, self.target_size)
                contrastive_images.append(positive_img)
                labels.append(is_positive)
            else:
                # Selecting negative sample from a different class
                negative_df = self.df[self.df['class'] != anchor_row['class']]
                negative_df = negative_df.sample(frac = 1)
                negative_row = negative_df.iloc[np.random.randint(0, len(negative_df))]
                negative_img = tf.io.read_file(negative_row['rgb'])
                negative_img = tf.image.decode_png(negative_img, channels=3)
                negative_img = tf.image.resize(negative_img, self.target_size)
                contrastive_images.append(negative_img)
                labels.append(is_positive)

        anchor_images = tf.stack(anchor_images) / 255.0
        contrastive_images = tf.stack(contrastive_images) / 255.0

        return [anchor_images, contrastive_images], labels

    def reset(self):
        self.df = self.df.sample(frac=1) if self.subset == 'training' else self.df
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.df))
        if self.subset == 'training':
            np.random.shuffle(self.indexes)


def load_data(config):
    full_df = limit_data(config, config.num_img_lim)

    train_generator = MultiModalDataGenerator(full_df, config, subset='training')
    validation_generator = MultiModalDataGenerator(full_df, config, subset='validation')

    return train_generator, validation_generator
