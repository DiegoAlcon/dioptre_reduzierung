import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import tensorflow as tf
from keras import layers
import pickle
import tensorflow as tf
import random
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
            fig, axs = plt.subplots(2, 4, figsize=(15, 15))

            for i in range(2):
                for j in range(4):
                    index = i * 4 + j
                    if index < len(self.images):
                        axs[i, j].imshow(self.images[index], cmap='gray')
                        axs[i, j].axis("off") 

            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            plt.show()

class NeuralNet:
    # Define a custom U-Net model
    def custom_unet_model(self, input_shape):
        
        inputs = tf.keras.Input(shape=input_shape)

        # Here goes the UNet defined by the Functional API Blog
        def double_conv_block(x, n_filters):

            # Conv2D then ReLU activation
            x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
            # Conv2D then ReLU activation
            x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)

            return x
        def downsample_block(x, n_filters):
            f = double_conv_block(x, n_filters)
            p = layers.MaxPool2D(2)(f)
            p = layers.Dropout(0.3)(p)

            return f, p
        def upsample_block(x, conv_features, n_filters):
            # upsample
            x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
            # concatenate
            x = layers.concatenate([x, conv_features])
            # dropout
            x = layers.Dropout(0.3)(x)
            # Conv2D twice with ReLU activation
            x = double_conv_block(x, n_filters)

            return x
            
        inputs = layers.Input(shape=input_shape)
        # encoder: contracting path - downsample
        # 1 - downsample
        f1, p1 = downsample_block(inputs, 64)
        # 2 - downsample
        f2, p2 = downsample_block(p1, 128)
        # 3 - downsample
        f3, p3 = downsample_block(p2, 256)
        # 4 - downsample
        f4, p4 = downsample_block(p3, 512)

        # 5 - bottleneck
        bottleneck = double_conv_block(p4, 1024)

        # decoder: expanding path - upsample
        # 6 - upsample
        u6 = upsample_block(bottleneck, f4, 512)
        # 7 - upsample
        u7 = upsample_block(u6, f3, 256)
        # 8 - upsample
        u8 = upsample_block(u7, f2, 128)
        # 9 - upsample
        u9 = upsample_block(u8, f1, 64)

        # outputs
        outputs = layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9) # 3 and softmax with SparseCategoricalCrossentropy or 1 and sigmoid with BinaryCrossentropy

        model = tf.keras.Model(inputs, outputs, name="U-Net")
        return model
    
if __name__ == "__main__":
    # Kleiner Rechner
    images_folder = r'C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\original'
    which_folder  = int(input('Enter 1 for bubble masking, 2 for gesamte masking, 3 for Differenz masking: '))
    if which_folder == 1:
        masks_folder = r'C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\bubbles'
    elif which_folder == 2:
        masks_folder = r'C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\volumen'
    elif which_folder == 3:
        masks_folder = r'C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\segm'

    # Mittlerer Rechner
    #images_folder = r'C:\Users\SANCHDI2\dioptre_reduzierung\original'
    #which_folder = int(input('Enter 1 for bubble masking, 2 for gesamte masking, 3 for Differenz masking: '))   
    #if which_folder == 1:
    #    masks_folder = r'C:\Users\SANCHDI2\dioptre_reduzierung\bubbles' 
    #elif which_folder == 2:
    #    masks_folder = r'C:\Users\SANCHDI2\dioptre_reduzierung\volumen'
    #elif which_folder == 3:
    #    masks_folder = r'C:\Users\SANCHDI2\dioptre_reduzierung\segm' 

    # Grösster Rechner
    #images_folder = r'C:\Users\SANCHDI2\OneDrive - Alcon\Desktop\Blasenentfernung\dioptre_reduzierung\original'
    #which_folder = int(input('Enter 1 for bubble masking, 2 for gesamte masking, 3 for Differenz masking: '))  
    #if which_folder == 1:
    #    masks_folder = r'C:\Users\SANCHDI2\OneDrive - Alcon\Desktop\Blasenentfernung\dioptre_reduzierung\bubbles'
    #elif which_folder == 2:
    #    masks_folder = r'C:\Users\SANCHDI2\OneDrive - Alcon\Desktop\Blasenentfernung\dioptre_reduzierung\volumen'
    #elif which_folder == 3:
    #    masks_folder = r'C:\Users\SANCHDI2\OneDrive - Alcon\Desktop\Blasenentfernung\dioptre_reduzierung\segm'

    images_files  = os.listdir(images_folder)
    masks_files = os.listdir(masks_folder)

    r = re.compile(r'\d+')
    images_files.sort(key=lambda x: int(r.search(x).group()))
    masks_files.sort(key=lambda x: int(r.search(x).group()))

    # The following lines are momentarily, and serve to highlight the importance of partitioning into differents stages of image aqcuisition
    # images_files = images_files[:] # write 96: to do only new images, :96 only old images, : all images
    # masks_files = masks_files[:] # write 96: to do only new images, :96 only old images, : all images

    new_height = 128
    new_width = 128

    with open("test", "rb") as fp:   
        diopts = pickle.load(fp)

    original_y = diopts

    y = list(filter(lambda x: -10 < x < 0, original_y))

    preserved_img = [1 if x in y else 0 for x in original_y]

    images = []
    masks = []

    for img_num, (image_file, mask_file) in enumerate(zip(images_files, masks_files)):
        if preserved_img[img_num]:
            # Read the image and mask
            image = cv2.imread(os.path.join(images_folder, image_file), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(os.path.join(masks_folder, mask_file), cv2.IMREAD_GRAYSCALE)

            # Resize the image and mask 
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            images.append(image)
            masks.append(mask)

    # Create an index list to shuffle
    num_images = len(images)
    index_list = list(range(num_images))

    # Shuffle the index list
    random.shuffle(index_list)

    # Initialize new shuffled lists for images and masks
    images = [images[i] for i in index_list]
    masks = [masks[i] for i in index_list]

    images  = [image  / 255 for image  in images]
    images = [tf.cast(image, dtype=tf.float32) for image in images]

    masks  = [mask  / 255 for mask  in masks ]
    masks = [tf.cast(mask, dtype=tf.float32) for mask in masks]

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

    ############################### - Train the Neural Network - #################################

    neural_net = NeuralNet()
    input_shape = (x_train[0].shape[0], x_train[0].shape[1], 1) 
    model = neural_net.custom_unet_model(input_shape)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])   

    model.summary()

    history = model.fit(x_train, y_train, epochs=50, batch_size=16, validation_data=(x_val, y_val))

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')  
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.show()

    # Predict masks for test images
    predicted_masks = model.predict(x_test)

    num_images = 8
    rows, cols = 2, 4

    fig, axs = plt.subplots(rows, cols, figsize=(15, 8))

    x_test = [ (img - img.min()) / (img.max()- img.min()) for img in x_test]

    for i in range(num_images):
        row_idx = i // cols
        col_idx = i % cols

        # Combine original image and predicted mask using bitwise AND
        combined_image = cv2.bitwise_and(x_test[i], x_test[i], mask=(predicted_masks[i][:,:,0] < 0.5).astype(np.uint8))

        #axs[row_idx, col_idx].imshow(overlay_image, cmap="gray")
        axs[row_idx, col_idx].imshow(combined_image, cmap="gray")
        axs[row_idx, col_idx].set_title(f"Image {i+1}")
        axs[row_idx, col_idx].axis("off")

    plt.tight_layout()
    plt.show()
    
    model.save("UNet_Segmen_size_128_DE_epoche_50_ordnung.keras") # This line should not be run

    print('Hello world') 