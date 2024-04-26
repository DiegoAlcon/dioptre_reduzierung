import os
import cv2
import tensorflow as tf
#import pywt
#import keras
#from keras import layers
import numpy as np
import pandas as pd
#from keras.utils import to_categorical
#from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
#from keras.callbacks import EarlyStopping
#from keras.optimizers import Adam
import matplotlib.pyplot as plt
#from scipy import stats
#from sklearn.preprocessing import MinMaxScaler
#from decimal import Decimal
#from keras.applications import VGG16
from keras.layers import Input, Flatten, Dense
#from keras.models import Model


class BildPlotter:
    def __init__(self, image_data):
        self.image_data = image_data

    def plot_image(self):
        fig, ax = plt.subplots()
        ax.imshow(self.image_data, cmap='gray')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title('Sample Image')
        plt.show()

class Bildvorverarbeitung:
    def __init__(self, image_directory, excel_directory, target_height, target_width, x_offset, y_offset): 
        self.image_directory = image_directory
        self.excel_directory = excel_directory                                                             
        self.target_height   = target_height                                    
        self.target_width    = target_width
        self.x_offset        = x_offset
        self.y_offset        = y_offset
        self.images          = self.load_images_from_directory_call()

    def load_images_from_directory_call(self):
        images, diopt = self.load_images_from_directory()
        return images, diopt
    
    def load_images_from_directory(self):
        images = []  
        diopt = []    

        df = pd.read_excel(self.excel_directory) 
        title_exc = df['Title'].tolist()
        sample_exc = df['Sample'].tolist()
        diopt_pre = df['OPTIC OF - Pre VD PB'].tolist()
        diopt_post = df['OPTIC OF - Post VD PB'].tolist()

        for directory in self.image_directory:
            directory_filenames = os.listdir(directory)
            for filename in directory_filenames:
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    base_name, ext = os.path.splitext(filename)
                    parts = base_name.split("_")

                    title_img = parts[2][:6]
                    sample_img = parts[2][7:]

                    idx_title = [i for i, a in enumerate(title_exc) if a == title_img]
                    idx_sample   = [i for i, a in enumerate(sample_exc) if a == int(sample_img)]

                    if len(idx_title) != 0 and len(idx_sample) != 0:
                        if len(np.intersect1d(idx_title, idx_sample)) != 0:
                            idx = np.intersect1d(idx_title, idx_sample)[0] 
                        else:
                            continue
                        if not(np.isnan(diopt_post[idx])) and not(np.isnan(diopt_pre[idx])):
                            diopt_delta = diopt_post[idx] - diopt_pre[idx]
                        else:
                            continue
                    else:
                        continue

                    images.append(cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE))
                    diopt.append(diopt_delta)

        return images, diopt
    
    def crop_images(self, images):
        cropped_images = []
        for img in images:
            if img.shape != (self.target_height, self.target_width):
                original_height, original_width = img.shape[:2]
                left = ((original_width - self.target_width) // 2 ) + self.x_offset
                top = ((original_height - self.target_height) // 2 ) + self.y_offset
                right = left + self.target_width 
                bottom = top + self.target_height 
                cropped_images.append(img[top:bottom, left:right]) 

            else:
                cropped_images.append(img) 
        return cropped_images
    
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
    def unsharp_mask(self, kernel_size=(11, 11), sigma=3, amount=4.0, threshold=10): # kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0
        sharpened_images = []
        for img in self.images:
            blurred = cv2.GaussianBlur(img, kernel_size, sigma)
            sharpened = int(amount + 1) * img - int(amount) * blurred # float was exchanged by an int
            sharpened_images.append(np.maximum(sharpened, np.zeros_like(img)))
        return sharpened_images

    # Sharpen Versuch 3
    def apply_canny(self):
        canny_images = []
        for img in self.images:
            canny_images.append(cv2.Canny(img, 10, 300, L2gradient=True, apertureSize=7))
        return canny_images
    
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
            fig, axs = plt.subplots(8, 11, figsize=(15, 15))

            for i in range(8):
                for j in range(11):
                    index = i * 11 + j
                    if index < len(self.images):
                        axs[i, j].imshow(self.images[index], cmap='gray')
                        axs[i, j].axis("off") 

            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            plt.show()
    
if __name__ == "__main__":
    # For the first computer
    #image_directory = [r"C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\Labeled1", 
    #                   r"C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\Labeled2", 
    #                   ]  
    # For the second computer (grössest)
    # For the third computer (von Timo (Mittlere))
    image_directory = [r"C:\Users\SANCHDI2\dioptre_reduzierung\Labeled1",
                       r"C:\Users\SANCHDI2\dioptre_reduzierung\Labeled2"
                        ]
    excel_directory = "example.xlsx"
    image_processor = Bildvorverarbeitung(image_directory, excel_directory, target_height=850, target_width=850, x_offset=-225, y_offset=1250)

    images = image_processor.images[0]
    diopts = image_processor.images[1]

    images = image_processor.crop_images(images)

    #sharpen_image = Merkmalsextraktion(images) 
    #images = sharpen_image.unsharp_mask()

    #highpass_image = Merkmalsextraktion(images)
    #images = highpass_image.highpass_sharpen()

    canny_edged_image = Merkmalsextraktion(images)
    images = canny_edged_image.apply_canny()

    images = [image / 255 for image in images]

    #image_plotter = BildPlotter(images) 
    #image_plotter.plot_image(2) # 1 soll images index werden, 2 darf es nicht

    factor = 4

    new_height = images[0].shape[0] // factor
    new_width   = images[0].shape[1] // factor

    images = [cv2.resize(img, (new_height, new_width), interpolation=cv2.INTER_AREA) for img in images]

    #image_plotter = BildPlotter(images) 
    #image_plotter.plot_image(2) # 1 soll images index werden, 2 darf es nicht

    del images[55]
    del diopts[55]

    x = images
    y = diopts

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    x_train = np.array(x_train)
    x_val   = np.array(x_val)
    x_test  = np.array(x_test)

    y_train = np.array(y_train)
    y_val   = np.array(y_val)
    y_test  = np.array(y_test)

    model_zu_nutzen = 2

    if model_zu_nutzen == 1:
        input_layer = tf.keras.Input(shape=(x_train.shape[1], x_train.shape[2], 1))
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

    elif model_zu_nutzen == 2:
        input_layer = Input(shape=(x_train.shape[1], x_train.shape[2], 1))
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        output_layer = tf.keras.layers.Dense(1, activation='linear')(x)

    elif model_zu_nutzen == 3:
        input_layer = Input(shape=(x_train.shape[1], x_train.shape[2], 1))
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        output_layer = tf.keras.layers.Dense(1, activation='linear')(x)

    # Create the model
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

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