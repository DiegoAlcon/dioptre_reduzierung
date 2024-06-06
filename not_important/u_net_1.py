import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras import layers

#############################################################################################################################################################################

# Load your image (replace 'your_image.jpg' with the actual image file)
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

##############################################################################################################################################################################

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

##############################################################################################################################################################################

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
    
##############################################################################################################################################################################

class NeuralNet:
    # Define the U-Net model
    def unet_model(self, input_shape):
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
        concat4 = layers.concatenate([conv2, up4], axis=-1) # fehler
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
    
##############################################################################################################################################################################

if __name__ == "__main__":
    # Klein Rechner
    #image_directory = [r"C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\Labeled1", 
    #                   r"C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\Labeled2", 
    #                   ]  
    # Mittlere Rechner
    #image_directory = [r"C:\Users\SANCHDI2\dioptre_reduzierung\Labeled1",
    #                   r"C:\Users\SANCHDI2\dioptre_reduzierung\Labeled2"
    #                    ]
     # For riesiger Rechner:
    image_directory = [r"C:\Users\SANCHDI2\OneDrive - Alcon\Desktop\Blasenentfernung\dioptre_reduzierung\Labeled1",
                       r"C:\Users\SANCHDI2\OneDrive - Alcon\Desktop\Blasenentfernung\dioptre_reduzierung\Labeled2",
                       ]
    excel_directory = "example.xlsx"
    image_processor = Bildvorverarbeitung(image_directory, excel_directory, target_height=900, target_width=900, x_offset=-225, y_offset=1250)

    images = image_processor.images[0]
    diopts = image_processor.images[1]

    images = image_processor.crop_images(images)

    #canny_edged_image = Merkmalsextraktion(images)
    #images = canny_edged_image.apply_canny()

    #images = [image / 255 for image in images]

    del images[55]
    del diopts[55]

    factor = 3
    new_height = images[0].shape[0] // factor
    new_width   = images[0].shape[1] // factor
    images = [cv2.resize(img, (new_height, new_width), interpolation=cv2.INTER_AREA) for img in images]

    x = images
    y = diopts

    train_size = int(0.8 * len(x))
    x_train, x_temp = x[:train_size], x[train_size:]
    test_size = int(0.5 * len(x_temp))
    x_val, x_test = x_temp[:test_size], x_temp[test_size:]

   # Mittlere Rechner
   #folder_path = r"C:\Users\SANCHDI2\dioptre_reduzierung\masks"
   # Riesiger Rechner
    folder_path = r"C:\Users\SANCHDI2\OneDrive - Alcon\Desktop\Blasenentfernung\dioptre_reduzierung\masks"
    x_masks = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_array is not None:
                x_masks.append(img_array)

    new_height_masks = x_masks[0].shape[0] // factor
    new_width_masks   = x_masks[0].shape[1] // factor
    x_masks = [cv2.resize(mask, (new_height_masks, new_width_masks), interpolation=cv2.INTER_AREA) for mask in x_masks]

    x_train_masks = x_masks[:69]
    x_val_masks   = x_masks[69:78]
    x_test_masks  = x_masks[78:]

    x_train = np.array(x_train)
    x_val   = np.array(x_val)
    x_test  = np.array(x_test)

    x_train_masks = np.array(x_train_masks)
    x_val_masks   = np.array(x_val_masks)
    x_test_masks  = np.array(x_test_masks)

    # Create thU-Net model
    input_shape = (x[0].shape[0], x[0].shape[1], 1)  
    #num_images  = len(x_train)
    neural_net = NeuralNet()
    model = neural_net.unet_model(input_shape)

    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

    # Train the model
    model.fit(x_train, x_train_masks, epochs=10, batch_size=16, validation_data=(x_val, x_val_masks), callbacks=[early_stopping])

    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(x_test, x_test_masks)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Predict masks for test images
    test_predictions = model.predict(x_test)

    # Color pixels predicted to belong to the object with white (255)
    test_predictions[test_predictions >= 0.5] = 255

    # Count total pixels predicted to belong to the object
    total_predicted_pixels = int((test_predictions > 0).sum())
    print(f"Total predicted pixels: {total_predicted_pixels}")

    print('Hello world')