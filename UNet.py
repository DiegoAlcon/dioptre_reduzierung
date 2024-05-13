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
        type_net = int(input('Enter 1 for 2-deep-2-layered, enter 2 for 2-deep-1-layered, enter 3 for less deep model: '))
        inputs = tf.keras.Input(shape=input_shape)

        if type_net == 1:
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

        elif type_net == 2:

            # Encoder (downsampling)
            conv1 = layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)
            pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

            conv2 = layers.Conv2D(128, 3, activation="relu", padding="same")(pool1)
            pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

            # Middle (bottleneck)
            conv3 = layers.Conv2D(256, 3, activation="relu", padding="same")(pool2)

            # Decoder (upsampling)
            up4 = layers.UpSampling2D(size=(2, 2))(conv3)
            concat4 = layers.concatenate([conv2, up4], axis=-1)
            conv4 = layers.Conv2D(128, 3, activation="relu", padding="same")(concat4)

            up5 = layers.UpSampling2D(size=(2, 2))(conv4)
            concat5 = layers.concatenate([conv1, up5], axis=-1)
            conv5 = layers.Conv2D(64, 3, activation="relu", padding="same")(concat5)

            # Output layer
            outputs = layers.Conv2D(1, 1, activation="sigmoid")(conv5)

        elif type_net == 3:
            # Encoder (downsampling)
            conv1 = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
            pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

            conv2 = layers.Conv2D(64, 3, activation="relu", padding="same")(pool1)
            pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

            # Middle (bottleneck)
            conv3 = layers.Conv2D(128, 3, activation="relu", padding="same")(pool2)

            # Decoder (upsampling)
            up4 = layers.UpSampling2D(size=(2, 2))(conv3)
            concat4 = layers.concatenate([conv2, up4], axis=-1)
            conv4 = layers.Conv2D(64, 3, activation="relu", padding="same")(concat4)

            up5 = layers.UpSampling2D(size=(2, 2))(conv4)
            concat5 = layers.concatenate([conv1, up5], axis=-1)
            conv5 = layers.Conv2D(32, 3, activation="relu", padding="same")(concat5)

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

        # Extract specific layers from VGG16 for skip connections
        encoder_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']
        encoder_outputs = [base_model.get_layer(layer_name).output for layer_name in encoder_layers]

        conv1 = layers.Conv2D(512, 3, activation='relu', padding='same')(encoder_outputs[-1])
        up1 = layers.UpSampling2D((2, 2))(conv1)
        concat1 = layers.concatenate([encoder_outputs[-2], up1], axis=-1)

        conv2 = layers.Conv2D(256, 3, activation='relu', padding='same')(concat1)
        up2 = layers.UpSampling2D((2, 2))(conv2)
        concat2 = layers.concatenate([encoder_outputs[-3], up2], axis=-1)

        up3 = layers.UpSampling2D((2, 2))(concat2)
        concat3 = layers.concatenate([encoder_outputs[-4], up3], axis=-1)

        # Output layer (binary mask)
        outputs = layers.Conv2D(3, 1, activation='softmax')(concat3)    
        # Create the U-Net model
        model = Model(inputs, outputs)
        return model
    
if __name__ == "__main__":
    # Kleiner Rechner
    images_folder = r'C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\original'
    which_folder = int(input('Enter 1 for bubbles masking, 2 for Gesamte masking, 3 for Difference masking: '))
    if which_folder == 1:
        masks_folder = r'C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\bubbles'
    elif which_folder == 2:
        masks_folder = r'C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\volumen'
    elif which_folder == 3:
        masks_folder = r'C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\segm'
    # Mittlerer Rechner
    #images_folder = r'C:\Users\SANCHDI2\dioptre_reduzierung\original'
    #masks_folder = r'C:\Users\SANCHDI2\dioptre_reduzierung\bubbles'
    
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

    y = list(filter(lambda x: -10 < x < 0, original_y))

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

    #bild = BildPlotter(mask)
    #bild.plot_image(1)
    
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
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['mean_absolute_error', 'mean_squared_error', 'accuracy'])

    model.summary()

    #early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

    # Train the model
    #history = model.fit(x_train, y_train, epochs=20, batch_size=16, validation_data=(x_val, y_val), callbacks=[early_stopping])
    history = model.fit(x_train, y_train, epochs=30, batch_size=16, validation_data=(x_val, y_val))

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')  
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.show()

    # Predict masks for test images
    predicted_masks = model.predict(x_test)

    if type_model == 1:
        # Remove the third dimension from predicted masks
        predicted_masks = np.squeeze(predicted_masks, axis=-1)
    elif type_model == 2:

        def rgb_to_gray(rgb_array):
            return np.dot(rgb_array[..., :3], [0.2989, 0.5870, 0.1140])
    
        predicted_masks = [rgb_to_gray(predicted_mask) for predicted_mask in predicted_masks]
        x_test = [rgb_to_gray(x_test_single) for x_test_single in x_test]

    # Create subplots
    num_images = len(x_test)
    rows, cols = 2, 4

    fig, axs = plt.subplots(rows, cols, figsize=(15, 8))

    for i in range(num_images):
        row_idx = i // cols
        col_idx = i % cols

        # Overlay predicted mask on original image (grayscale)
        #overlay_image = np.copy(x_test[i])
        #overlay_image[predicted_masks[i] > 0.5] = 255  # Set mask regions to white

        # Combine original image and predicted mask using bitwise AND
        combined_image = cv2.bitwise_and(x_test[i], x_test[i], mask=(predicted_masks[i] < 0.5).astype(np.uint8))

        #axs[row_idx, col_idx].imshow(overlay_image, cmap="gray")
        axs[row_idx, col_idx].imshow(combined_image, cmap="gray")
        axs[row_idx, col_idx].set_title(f"Image {i+1}")
        axs[row_idx, col_idx].axis("off")

    plt.tight_layout()
    plt.show()

    print('Hello world')