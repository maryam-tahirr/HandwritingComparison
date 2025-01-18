import os
import numpy as np
import pandas as pd
import tf_keras as keras
from tf_keras.models import Model, load_model
from tf_keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tf_keras.optimizers import Adam
from tf_keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tkinter import Tk, filedialog, Label, Button
from PIL import Image, ImageTk

# Load the CSV file and define constants
data_path = "combined_output_pairs.csv"
image_folder = "2combined_output_images"
image_size = (128, 128)
batch_size = 32
epochs_per_batch = 10

# Data Generator
class SiameseDataGenerator(keras.utils.Sequence):
    def __init__(self, dataframe, image_folder, batch_size, image_size):
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.image_size = image_size
        self.indices = np.arange(len(dataframe))

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.dataframe.iloc[batch_indices]

        images1 = []
        images2 = []
        labels = []

        for _, row in batch_data.iterrows():
            img1_path = os.path.join(self.image_folder, row['Image1'])
            img2_path = os.path.join(self.image_folder, row['Image2'])

            if os.path.exists(img1_path) and os.path.exists(img2_path):
                img1 = img_to_array(load_img(img1_path, target_size=self.image_size)) / 255.0
                img2 = img_to_array(load_img(img2_path, target_size=self.image_size)) / 255.0

                images1.append(img1)
                images2.append(img2)
                labels.append(row['Label'])

        return [np.array(images1), np.array(images2)], np.array(labels)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# Define the Siamese Network
def build_siamese_model():
    input_shape = image_size + (3,)

    def create_base_network(input_shape):
        input = Input(shape=input_shape)
        x = Conv2D(64, (3, 3), activation='relu')(input)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        return Model(input, x)

    base_network = create_base_network(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # Compute the absolute difference between the two encodings
    distance = Lambda(lambda tensors: keras.backend.abs(tensors[0] - tensors[1]))([processed_a, processed_b])

    output = Dense(1, activation='sigmoid')(distance)

    model = Model([input_a, input_b], output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])

    return model

# Load the data
data = pd.read_csv(data_path)

# Split into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create data generators
train_generator = SiameseDataGenerator(train_data, image_folder, batch_size, image_size)
test_generator = SiameseDataGenerator(test_data, image_folder, batch_size, image_size)

# Build and train the model
model = build_siamese_model()

model.fit(train_generator, validation_data=test_generator, epochs=epochs_per_batch)
model.save("siamese_model.h5")


# Ensure the model doesn't train again for testing
model = load_model("siamese_model.h5")