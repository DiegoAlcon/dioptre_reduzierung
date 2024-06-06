# This program aims to implement a multilayer perceptron

import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import pickle
from keras.callbacks import EarlyStopping
import re

class BildPlotter:
    def __init__(self, images):
        self.images = images

    def plot_image(self, option):
        if option == 1:
            fig, ax = plt.subplots()
            ax.imshow(self.images, cmap='gray')
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_title('Sample Image')
            plt.show()
        elif option == 2:
            fig, axs = plt.subplots(8, 12, figsize=(15, 15))

            for i in range(8):
                for j in range(12):
                    index = i * 12 + j
                    if index < len(self.images):
                        axs[i, j].imshow(self.images[index], cmap='gray')
                        axs[i, j].axis("off") 

            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            plt.show()

# Define the paths to the original images folder and the masks folder
bubble_masks_folder    = r"C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\bubbles"
volume_masks_folder    = r"C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\volumen"
segmen_masks_folder    = r"C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\segm"

# Get the list of image files in both folders
bubble_masks_files    = os.listdir(bubble_masks_folder)
volume_masks_files    = os.listdir(volume_masks_folder)
segmen_masks_files    = os.listdir(segmen_masks_folder)

r = re.compile(r'\d+')
bubble_masks_files.sort(key=lambda x: int(r.search(x).group()))
volume_masks_files.sort(key=lambda x: int(r.search(x).group()))
segmen_masks_files.sort(key=lambda x: int(r.search(x).group()))

factor = int(input('Enter factor for downsamplig (possibe options: 10, 12, 15, 18, 20): '))
new_height = 4500 // factor
new_width = 4500 // factor
original_height = new_height * factor
original_width = new_width * factor

with open("test", "rb") as fp:   
    diopts = pickle.load(fp)

original_y = diopts

y = list(filter(lambda x: -10 < x < 0, original_y))

preserved_img = [1 if x in y else 0 for x in original_y]

img_num = 0
bubble_mask_images = []
for mask_file in bubble_masks_files:
    if preserved_img[img_num]:
        bubble_mask_image = cv2.imread(os.path.join(bubble_masks_folder, mask_file), cv2.IMREAD_GRAYSCALE)
        bubble_mask_image = cv2.resize(bubble_mask_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        bubble_mask_images.append(bubble_mask_image)
    img_num += 1

img_num = 0
volume_mask_images = []
for mask_file in volume_masks_files:
    if preserved_img[img_num]:
        volume_mask_image = cv2.imread(os.path.join(volume_masks_folder, mask_file), cv2.IMREAD_GRAYSCALE)
        volume_mask_image = cv2.resize(volume_mask_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        volume_mask_images.append(volume_mask_image)
    img_num += 1

img_num = 0
segmen_mask_images = []
for mask_file in segmen_masks_files:
    if preserved_img[img_num]:
        segmen_mask_image = cv2.imread(os.path.join(segmen_masks_folder, mask_file), cv2.IMREAD_GRAYSCALE)
        segmen_mask_image = cv2.resize(segmen_mask_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        segmen_mask_images.append(segmen_mask_image)
    img_num += 1

x_abs_size_bubbles = []
for mask in bubble_mask_images: 
    abs_size = 0
    for row in mask:
        row_sum = sum(1 for value in row if value == 255)
        abs_size += row_sum
    x_abs_size_bubbles.append(abs_size)
#x_abs_size_bubbles = (x_abs_size_bubbles - np.mean(x_abs_size_bubbles)) / np.std(x_abs_size_bubbles) # Normalization
#x_abs_size_bubbles = (x_abs_size_bubbles - np.min(x_abs_size_bubbles))/(np.max(x_abs_size_bubbles) - np.min(x_abs_size_bubbles))

x_abs_size_volume = []
for mask in volume_mask_images: 
    abs_size = 0
    for row in mask:
        row_sum = sum(1 for value in row if value == 255)
        abs_size += row_sum
    x_abs_size_volume.append(abs_size)
#x_abs_size_volume = (x_abs_size_volume - np.mean(x_abs_size_volume)) / np.std(x_abs_size_volume) # Normalization
#x_abs_size_volume = (x_abs_size_volume - np.min(x_abs_size_volume))/(np.max(x_abs_size_volume) - np.min(x_abs_size_volume))

x_abs_size_segmen = []
for mask in segmen_mask_images: 
    abs_size = 0
    for row in mask:
        row_sum = sum(1 for value in row if value == 255)
        abs_size += row_sum
    x_abs_size_segmen.append(abs_size)

train_size_bubbles = int(0.8 * len(x_abs_size_bubbles))
x_train_bubbles, x_temp_bubbles = x_abs_size_bubbles[:train_size_bubbles], x_abs_size_bubbles[train_size_bubbles:]
test_size_bubbles = int(0.5 * len(x_temp_bubbles))
x_val_bubbles, x_test_bubbles = x_temp_bubbles[:test_size_bubbles], x_temp_bubbles[test_size_bubbles:]

train_size_volume = int(0.8 * len(x_abs_size_volume))
x_train_volume, x_temp_volume = x_abs_size_volume[:train_size_volume], x_abs_size_volume[train_size_volume:]
test_size_volume = int(0.5 * len(x_temp_volume))
x_val_volume, x_test_volume = x_temp_volume[:test_size_volume], x_temp_volume[test_size_volume:]

train_size_segmen = int(0.8 * len(x_abs_size_segmen))
x_train_segmen, x_temp_segmen = x_abs_size_segmen[:train_size_segmen], x_abs_size_segmen[train_size_segmen:]
test_size_segmen = int(0.5 * len(x_temp_segmen))
x_val_segmen, x_test_segmen = x_temp_segmen[:test_size_segmen], x_temp_segmen[test_size_segmen:]

train_size_dioptre = int(0.8 * len(y))
y_train_dioptre, y_temp_dioptre = y[:train_size_dioptre], y[train_size_dioptre:]
test_size_dioptre = int(0.5 * len(y_temp_dioptre))
y_val_dioptre, y_test_dioptre = y_temp_dioptre[:test_size_dioptre], y_temp_dioptre[test_size_dioptre:]

X_train = np.array([x_train_bubbles, x_train_segmen]).T # ABWECHSELND
Y_train = np.array(y_train_dioptre).reshape(-1, 1)

X_val   = np.array([x_val_bubbles, x_val_segmen]).T # ABWECHSELND
Y_val   = np.array(y_val_dioptre).reshape(-1, 1)

X_test   = np.array([x_test_bubbles, x_test_segmen]).T # ABWECHSELND
Y_test   = np.array(y_test_dioptre).reshape(-1, 1)
#X_test   = np.array([x_abs_size_bubbles, x_abs_size_volume, x_abs_size_segmen]).T
#Y_test   = np.array(y).reshape(-1, 1)

# Define an even more robust neural network model with additional techniques
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=512, activation='relu', input_shape=(2,)), # ABWECHSELND
    tf.keras.layers.Dropout(0.2),  # Add dropout layer for regularization
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.BatchNormalization(),  # Add batch normalization layer
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dropout(0.2),  # Add dropout layer for regularization
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1)  # Output layer (no activation function for regression)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])

model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1, restore_best_weights=True)

# Train the model
history = model.fit(X_train, Y_train, epochs=300, batch_size=16, validation_data=(X_val, Y_val), callbacks=[early_stopping]) 

#test_loss, test_mae = model.evaluate(X_test, Y_test)
#print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')  
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.show()

# Make predictions
predictions = model.predict(X_test)

plt.scatter(Y_test, predictions, alpha=0.5)
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='r', linestyle='--', label='Ideal Line')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Predictions vs. True Values')
plt.show()

# Print the predictions
#print("Predictions:")
#for i in range(len(predictions)):
#    print(f"Input: ({x_test_bubbles[i]}, {x_test_volume[i]}, {x_test_segmen[i]}), Predicted Output: {predictions[i][0]:.2f}")

print('Hello World')