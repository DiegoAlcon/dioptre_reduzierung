# This programs aims to create a CNN couppled with Transfer Learning algorithm for the dioptre prediction from images of IOLs

import os
import cv2
import pywt
import keras
from keras import layers
import numpy as np
import pandas as pd
import openpyxl
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from skimage.transform import rotate
import random
from keras.applications import VGG16
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

    factor = int(input('Enter factor for downsamplig (possibe options: 1, 2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20): '))
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

    #images = [cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA) for img in images]

    # Merkmalsextraktion
    #sharpen_image = Merkmalsextraktion(images) 
    #images = sharpen_image.unsharp_mask()

    #highpass_image = Merkmalsextraktion(images)
    #images = highpass_image.highpass_sharpen()

    #canny_edged_image = Merkmalsextraktion(images)
    #images = canny_edged_image.apply_canny()

    x = images
    y = diopts

    # x_augmented = []
    # y_augmented = []

    # for image, value in zip(x, y):
        
        # rotated_90 = np.rot90(image, k=90 // 90) # first argument: 90, 180, 270 / second argument: 90
        # rotated_180 = np.rot90(image, k=180 // 90)
        # rotated_270 = np.rot90(image, k=270 // 90)
    
        # x_augmented.extend([rotated_90, rotated_180, rotated_270])
        # y_augmented.extend([value, value, value])

    # x_combined = x + x_augmented
    # y_combined = y + y_augmented

    # combined_data = list(zip(x_combined, y_combined))
    # random.shuffle(combined_data)
    # x, y = zip(*combined_data)

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

    x_train = np.repeat(x_train[..., np.newaxis], 3, axis=-1)
    x_val = np.repeat(x_val[..., np.newaxis], 3, axis=-1)
    x_test = np.repeat(x_test[..., np.newaxis], 3, axis=-1)

    y_train = np.array(y_train)
    y_val   = np.array(y_val)
    y_test  = np.array(y_test)

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(new_height, new_width, 3)) 

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    output_layer = Dense(1, activation='linear')(x)  

    model = Model(inputs=base_model.input, outputs=output_layer)

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

    history = model.fit(x_train, y_train, epochs=10, batch_size=16, validation_data=(x_val, y_val), callbacks=[early_stopping]) 

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