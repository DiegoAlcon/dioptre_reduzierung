import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import pickle
import re
import random

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
            fig, axs = plt.subplots(9, 9, figsize=(15, 15))

            for i in range(9):
                for j in range(9):
                    index = i * 9 + j
                    if index < len(self.images):
                        axs[i, j].imshow(self.images[index], cmap='gray')
                        axs[i, j].axis("off") 

            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            plt.show()

model_bubble           = tf.keras.models.load_model('UNet_Bubbles_size_128_DE_epoche_50_ordnung.keras')
model_segmen           = tf.keras.models.load_model('UNet_Segmen_size_128_DE_epoche_50_ordnung.keras')

original_folder        = r'C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\original'
original_files         = os.listdir(original_folder)
r = re.compile(r'\d+')
original_files.sort(key=lambda x: int(r.search(x).group()))
new_height = 128
new_width = 128

with open("test", "rb") as fp:   
    diopts = pickle.load(fp)

original_y = diopts
y = list(filter(lambda x: -10 < x < 0, original_y))
original_y = -np.array(original_y)
y = -np.array(y)
preserved_img = [1 if x in y else 0 for x in original_y]

original_images = []
y_augmented = []
num_rotations = 0 # write 0 to not do DA

for img_num, (original_file, y_) in enumerate(zip(original_files, original_y)):
    if preserved_img[img_num]:
        original_image = cv2.imread(os.path.join(original_folder, original_file), cv2.IMREAD_GRAYSCALE)
        original_image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        for rotation_idx in range(num_rotations):
            angle = random.uniform(0, 360)
            M = cv2.getRotationMatrix2D((new_width // 2, new_height // 2), angle, 1)
            rotated_image = cv2.warpAffine(original_image, M, (new_width, new_height))
            original_images.append(rotated_image)
            y_augmented.append(y_)
        original_images.append(original_image)
        y_augmented.append(y_)

num_images = len(original_images)
index_list = list(range(num_images))
random.shuffle(index_list)
original_images = [original_images[i] for i in index_list]
y_augmented = [y_augmented[i] for i in index_list]

original_images  = [original_image  / 255 for original_image  in original_images]
original_images = [tf.cast(original_image, dtype=tf.float32) for original_image in original_images]
original_images = np.array(original_images)
y = np.array(y_augmented)

bubble_mask_images = model_bubble.predict(original_images)
segmen_mask_images = model_segmen.predict(original_images)

#whal = int(input('1: Volume is included, 0: volume is not included, enter: '))
whal = 0

x_abs_size_bubbles = []
x_abs_size_segmen = []
for bubble_mask, segmen_mask in zip(bubble_mask_images, segmen_mask_images):
    bubble_mask = bubble_mask[:, :, 0]
    segmen_mask = segmen_mask[:, :, 0]
    num_pixels_bubbles = np.sum(bubble_mask < 0.5) # threshold adjusted to Merkmalebedurfnisse
    num_pixels_segmen = np.sum(segmen_mask < 0.5) # threshold adjusted to Merkmalebedurfnisse
    x_abs_size_bubbles.append(num_pixels_bubbles)
    x_abs_size_segmen.append(num_pixels_segmen)

train_size_bubbles = int(0.8 * len(x_abs_size_bubbles))
x_train_bubbles, x_temp_bubbles = x_abs_size_bubbles[:train_size_bubbles], x_abs_size_bubbles[train_size_bubbles:]
test_size_bubbles = int(0.5 * len(x_temp_bubbles))
x_val_bubbles, x_test_bubbles = x_temp_bubbles[:test_size_bubbles], x_temp_bubbles[test_size_bubbles:]

train_size_segmen = int(0.8 * len(x_abs_size_segmen))
x_train_segmen, x_temp_segmen = x_abs_size_segmen[:train_size_segmen], x_abs_size_segmen[train_size_segmen:]
test_size_segmen = int(0.5 * len(x_temp_segmen))
x_val_segmen, x_test_segmen = x_temp_segmen[:test_size_segmen], x_temp_segmen[test_size_segmen:]

train_size_dioptre = int(0.8 * len(y))
y_train_dioptre, y_temp_dioptre = y[:train_size_dioptre], y[train_size_dioptre:]
test_size_dioptre = int(0.5 * len(y_temp_dioptre))
y_val_dioptre, y_test_dioptre = y_temp_dioptre[:test_size_dioptre], y_temp_dioptre[test_size_dioptre:]

if whal == 1:
    model_volume = tf.keras.models.load_model('UNet_volume256.keras')
    volume_mask_images = model_volume.predict(original_images)
    x_abs_size_volume = []
    for volume_mask in volume_mask_images:
        volume_mask = volume_mask[:, :, 0]
        num_pixels_volume = np.sum(volume_mask < 0.5) # threshold adjusted to Merkmalebedurfnisse
        x_abs_size_volume.append(num_pixels_volume)

    train_size_volume = int(0.8 * len(x_abs_size_volume))
    x_train_volume, x_temp_volume = x_abs_size_volume[:train_size_volume], x_abs_size_volume[train_size_volume:]
    test_size_volume = int(0.5 * len(x_temp_volume))
    x_val_volume, x_test_volume = x_temp_volume[:test_size_volume], x_temp_volume[test_size_volume:]

    X_train = np.array([x_train_bubbles, x_train_volume, x_train_segmen]).T
    Y_train = np.array(y_train_dioptre).reshape(-1, 1)

    X_val   = np.array([x_val_bubbles, x_val_volume, x_val_segmen]).T
    Y_val   = np.array(y_val_dioptre).reshape(-1, 1)

    X_test   = np.array([x_test_bubbles, x_test_volume, x_test_segmen]).T 
    Y_test   = np.array(y_test_dioptre).reshape(-1, 1)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=512, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dropout(0.2),  
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.BatchNormalization(),  
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dropout(0.2),  
        tf.keras.layers.Dense(units=64,  activation='relu'),
        tf.keras.layers.Dense(units=32,  activation='relu'),
        tf.keras.layers.Dense(units=1, activation='linear')  
    ])

elif whal == 0:

    X_train = np.array([x_train_bubbles, x_train_segmen]).T
    Y_train = np.array(y_train_dioptre).reshape(-1, 1)

    X_val   = np.array([x_val_bubbles, x_val_segmen]).T
    Y_val   = np.array(y_val_dioptre).reshape(-1, 1)

    X_test   = np.array([x_test_bubbles, x_test_segmen]).T 
    Y_test   = np.array(y_test_dioptre).reshape(-1, 1)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=512, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dropout(0.2),  
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.BatchNormalization(),  
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dropout(0.2),  
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='linear')  
    ])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])

model.summary()

history = model.fit(X_train, Y_train, epochs=600, batch_size=16, validation_data=(X_val, Y_val)) 

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')  
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.show()

predictions = model.predict(X_test)

plt.scatter(Y_test, predictions, alpha=0.5)
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='r', linestyle='--', label='Ideal Line')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Predictions vs. True Values')
plt.show()

model.save('MLP_epoche_600_size_128_ordnung.keras')

print('Hello World')