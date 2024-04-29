# The following code aims to compute a U-Net simplification for automatic masking purposes. It unites Custom as well as TL implementation.

import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras import layers
import pywt
import keras
from keras import layers
import openpyxl
from keras.callbacks import EarlyStopping
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from scipy import fftpack, ndimage
import pickle
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from keras import backend as K

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

class NeuralNet:
    # Define a custom U-Net model
    def custom_unet_model(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        #inputs = [tf.keras.Input(shape=input_shape) for _ in range(num_images)]
        #inputs = layers.concatenate(inputs, axis=-1)

        # Encoder (downsampling)
        conv1 = layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)
        conv1 = layers.Conv2D(64, 3, activation="relu", padding="same")(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = layers.Conv2D(128, 3, activation="relu", padding="same")(pool1)
        conv2 = layers.Conv2D(128, 3, activation="relu", padding="same")(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        # Middle (bottleneck)
        conv3 = layers.Conv2D(256, 3, activation="relu", padding="same")(pool2)
        conv3 = layers.Conv2D(256, 3, activation="relu", padding="same")(conv3)

        # Decoder (upsampling)
        up4 = layers.UpSampling2D(size=(2, 2))(conv3)
        concat4 = layers.concatenate([conv2, up4], axis=-1)
        conv4 = layers.Conv2D(128, 3, activation="relu", padding="same")(concat4)
        conv4 = layers.Conv2D(128, 3, activation="relu", padding="same")(conv4)

        up5 = layers.UpSampling2D(size=(2, 2))(conv4)
        concat5 = layers.concatenate([conv1, up5], axis=-1)
        conv5 = layers.Conv2D(64, 3, activation="relu", padding="same")(concat5)
        conv5 = layers.Conv2D(64, 3, activation="relu", padding="same")(conv5)

        # Output layer
        outputs = layers.Conv2D(1, 1, activation="sigmoid")(conv5)

        model = tf.keras.Model(inputs, outputs)
        return model
    
    def TL_unet_model(self, input_shape):
        ## Load pre-trained VGG16 model (excluding top layers)
        #vgg16_base = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
        ## Freeze the pre-trained layers
        #for layer in vgg16_base.layers:
        #    layer.trainable = False
        ## Get the output from the last convolutional layer of VGG16
        #encoder_output = vgg16_base.get_layer("block5_conv3").output
        ## Decoder (upsampling)
        #up4 = layers.UpSampling2D(size=(2, 2))(encoder_output)
        #concat4 = layers.concatenate([vgg16_base.get_layer("block4_conv3").output, up4], axis=-1)
        #conv4 = layers.Conv2D(128, 3, activation="relu", padding="same")(concat4)
        #conv4 = layers.Conv2D(128, 3, activation="relu", padding="same")(conv4)
        #up5 = layers.UpSampling2D(size=(2, 2))(conv4)
        #concat5 = layers.concatenate([vgg16_base.get_layer("block3_conv3").output, up5], axis=-1)
        #conv5 = layers.Conv2D(64, 3, activation="relu", padding="same")(concat5)
        #conv5 = layers.Conv2D(64, 3, activation="relu", padding="same")(conv5)
        ## Output layer
        #outputs = layers.Conv2D(1, 1, activation="sigmoid")(conv5)
        #model = tf.keras.Model(inputs=vgg16_base.input, outputs=outputs)
        ###################################################################################
        # Define input layer
        inputs = Input(input_shape)    
        # Use VGG16 as encoder (pre-trained on ImageNet)
        base_model = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)    
        # Freeze the layers in the base model
        for layer in base_model.layers:
            layer.trainable = False    
        # Decoder (upsampling path)
        # Add upsampling layers and skip connections
        # Note: You can customize the number of filters and kernel sizes
        # Example architecture:
        conv1 = layers.Conv2D(512, 3, activation='relu', padding='same')(base_model.output)
        up1 = layers.UpSampling2D((2, 2))(conv1)
        concat1 = layers.concatenate([base_model.get_layer('block4_pool').output, up1], axis=-1)    
        conv2 = layers.Conv2D(256, 3, activation='relu', padding='same')(concat1)
        up2 = layers.UpSampling2D((2, 2))(conv2)
        concat2 = layers.concatenate([base_model.get_layer('block3_pool').output, up2], axis=-1)    
        # Add more upsampling layers and skip connections as needed    
        # Output layer (binary mask)
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(concat2)    
        # Create the U-Net model
        model = Model(inputs, outputs)
        return model
    
if __name__ == "__main__":
    images_folder = r'C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\original'
    masks_folder = r'C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\bubbles'
    images_files  = os.listdir(images_folder)
    masks_files = os.listdir(masks_folder)

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

    img_num = 0
    masks = []
    for file in masks_files:
        if preserved_img[img_num]:
            mask = cv2.imread(os.path.join(masks_folder, file), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            masks.append(mask)
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
    y = masks

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

    neural_net = NeuralNet()
    type_model = int(input('Enter 1 for custom UNet, 2 for TL UNet: '))
    if type_model == 1:
        input_shape = (x_train[0].shape[0], x_train[0].shape[1], 1) 
        model = neural_net.custom_unet_model(input_shape)
    elif type_model == 2:
        x_train = np.repeat(x_train[..., np.newaxis], 3, axis=-1)
        x_val = np.repeat(x_val[..., np.newaxis], 3, axis=-1)
        x_test = np.repeat(x_test[..., np.newaxis], 3, axis=-1)
        y_train = np.repeat(y_train[..., np.newaxis], 3, axis=-1)
        y_val = np.repeat(y_val[..., np.newaxis], 3, axis=-1)
        y_test = np.repeat(y_test[..., np.newaxis], 3, axis=-1)
        input_shape = (x_train[0].shape[0], x_train[0].shape[1], 3) 
        model = neural_net.TL_unet_model(input_shape)

    # First definition of the dice coef loss func
    def dice_coefficient(y_true, y_pred):
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        return (2.0 * intersection + 1e-5) / (union + 1e-5)
    
    # Second definition of the dice coef loss func
    def dice_coef(y_true, y_pred, smooth=1):
        # flatten
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        # one-hot encoding y with 3 labels : 0=background, 1=label1, 2=label2
        y_true_f = K.one_hot(K.cast(y_true_f, np.uint8), 3)
        y_pred_f = K.one_hot(K.cast(y_pred_f, np.uint8), 3)
        # calculate intersection and union exluding background using y[:,1:]
        intersection = K.sum(y_true_f[:,1:]* y_pred_f[:,1:], axis=[-1])
        union = K.sum(y_true_f[:,1:], axis=[-1]) + K.sum(y_pred_f[:,1:], axis=[-1])
        # apply dice formula
        dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
        return dice
    def dice_loss(y_true, y_pred):
        return 1-dice_coef(y_true, y_pred)

    # Compile the model
    model.compile(optimizer="adam", loss=dice_loss, metrics=["accuracy"])

    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

    # Train the model
    history = model.fit(x_train, y_train, epochs=10, batch_size=16, validation_data=(x_val, y_val), callbacks=[early_stopping])

    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Predict masks for test images
    test_predictions = model.predict(x_test)

    # Color pixels predicted to belong to the object with white (255)
    test_predictions[test_predictions >= 0.5] = 255

    # Count total pixels predicted to belong to the object
    total_predicted_pixels = int((test_predictions > 0).sum())
    print(f"Total predicted pixels: {total_predicted_pixels}")

    print('Hello world')

    #test_loss, test_mae = model.evaluate(X_test, Y_test)
    #print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

    #plt.plot(history.history['loss'], label='train')
    #plt.plot(history.history['val_loss'], label='validation')  
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #plt.title('Training and Validation Loss Curves')
    #plt.legend()
    #plt.show()

    # Make predictions
    #predictions = model.predict(X_test)

    #plt.scatter(Y_test, predictions, alpha=0.5)
    #plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='r', linestyle='--', label='Ideal Line')
    #plt.xlabel('True Values')
    #plt.ylabel('Predictions')
    #plt.title('Predictions vs. True Values')
    #plt.show()

    # Print the predictions
    #print("Predictions:")
    #for i in range(len(predictions)):
        #print(f"Input: ({x_test_bubbles[i]}, {x_test_volume[i]}), Predicted Output: {predictions[i][0]:.2f}")

    #print('Hello World')