# This program aims to compute a Convolutional Neural Network to predict the change of the dioptre in images of IOLs

import os
import cv2
import pywt
import keras
from keras import layers
import numpy as np
import pandas as pd
import openpyxl
import tensorflow as tf
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from scipy import fftpack, ndimage
import pickle

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

class Merkmalsextraktion:
    def __init__(self, images):
        self.images = images

    # Sharpen Versuch 1
    def highpass_sharpen(self, kernel_size=7, alpha=2.0):
        highpassed_images = []
        for img in self.images:
            laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=kernel_size, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            highpassed_images_uncliped = img + alpha * laplacian
            highpassed_images.append(np.clip(highpassed_images_uncliped, 0, 255).astype(np.uint8))
        return highpassed_images
    
    # Sharpen Versuch 2
    def unsharp_mask(self, kernel_size=(11, 11), sigma=3, amount=4.0, threshold=10): 
        sharpened_images = []
        for img in self.images:
            blurred = cv2.GaussianBlur(img, kernel_size, sigma)
            sharpened = int(amount + 1) * img - int(amount) * blurred 
            sharpened_images.append(np.maximum(sharpened, np.zeros_like(img)))
        return sharpened_images

    # Sharpen Versuch 3
    def apply_canny(self):
        canny_images = []
        for img in self.images:
            canny_images.append(cv2.Canny(img, 10, 300, L2gradient=True, apertureSize=7))
        return canny_images
    
if __name__ == "__main__":
    
    images_folder = r'C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\original'
    images_files  = os.listdir(images_folder)

    factor = int(input('Enter factor for downsamplig (possibe options: 10, 12, 15, 18, 20): '))
    new_height = 4500 // factor
    new_width = 4500 // factor
    original_height = new_height * factor
    original_width = new_width * factor

    with open("test", "rb") as fp:   
        diopts = pickle.load(fp)

    original_y = diopts

    y = list(filter(lambda x: -5 < x < 5, original_y))

    preserved_img = [1 if x in y else 0 for x in original_y]

    img_num = 0
    images = []
    for file in images_files:
        if preserved_img[img_num]:
            image = cv2.imread(os.path.join(images_folder, file), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            images.append(image)
        img_num += 1

    images = [image / 255 for image in images]

    # Merkmalsextraktion
    #sharpen_image = Merkmalsextraktion(images) 
    #images = sharpen_image.unsharp_mask()

    #highpass_image = Merkmalsextraktion(images)
    #images = highpass_image.highpass_sharpen()

    #canny_edged_image = Merkmalsextraktion(images)
    #images = canny_edged_image.apply_canny()

    x = images
    y = diopts

    train_size_x = int(0.8 * len(x))
    x_train, x_temp = x[:train_size_x], x[train_size_x:]
    test_size_x = int(0.5 * len(x_temp))
    x_val, x_test = x_temp[:test_size_x], x_temp[test_size_x:]

    train_size_y = int(0.8 * len(y))
    y_train, y_temp = y[:train_size_y], y[train_size_y:]
    test_size_y = int(0.5 * len(y_temp))
    y_val, y_test = y_temp[:test_size_y], y_temp[test_size_y:]

    x_train = np.array(x_train)
    x_val   = np.array(x_val)
    x_test  = np.array(x_test)

    y_train = np.array(y_train)
    y_val   = np.array(y_val)
    y_test  = np.array(y_test)

    # Create a custom VGG16-like architecture
    input_layer = Input(shape=(new_height, new_width, 1))
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(1, activation='linear')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

    history = model.fit(x_train, y_train, epochs=10, batch_size=8, validation_data=(x_val, y_val), callbacks=[early_stopping]) 

    test_loss, test_mae = model.evaluate(x_test, y_test)
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')  
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.show()

    y_pred_test = model.predict(x_test)

    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='r', linestyle='--', label='Ideal Line')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs. True Values')
    plt.show()

    print('Hello World')