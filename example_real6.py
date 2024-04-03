import os
import cv2
import pywt
import keras
from keras import layers
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.utils import to_categorical
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from decimal import Decimal
#import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

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

class BildCirclesPlotter:
    def __init__(self, image, circles): 
        self.image = image
        self.circles = circles

    def plot_circles(self):
        if self.circles is not None:
            self.circles = np.uint16(np.around(self.circles))
            for circle in self.circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                cv2.circle(self.image, center, radius, (0, 0, 255), 2)

            plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            plt.title("Detected Circles")
            plt.axis("off")
            plt.show()
        else:
            print("No circles detected in the image.")

class Bildvorverarbeitung:
    def __init__(self, image_directory, target_height=750, target_width=750): # target_height=4504 target_width=4504
        self.image_directory = image_directory                                  # target_height=650, target_width=650
        self.target_height   = target_height                                    
        self.target_width    = target_width
        self.images          = self.load_images_from_directory_call()

    def load_images_from_directory_call(self):
        images1, diopt1 = self.load_images_from_directory1()
        images2, diopt2 = self.load_images_from_directory2()
        images3, diopt3 = self.load_images_from_directory3()
        images4, diopt4 = self.load_images_from_directory4()
        images5, diopt5 = self.load_images_from_directory5()
        images1.extend(images2)
        images1.extend(images3)
        images1.extend(images4)
        images1.extend(images5)
        diopt1.extend(diopt2)
        diopt1.extend(diopt3)
        diopt1.extend(diopt4)
        diopt1.extend(diopt5)
        return images1, diopt1
    
    def load_images_from_directory1(self):
        images = []  
        diopt = []   
        key_list = []  

        for filename in os.listdir(self.image_directory[0]):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                base_name, ext = os.path.splitext(filename)
                parts = base_name.split("_")

                blasen = parts[1] == 'after'

                key = int(parts[0][9:11])

                num_str = parts[-1].replace(",", ".")
                num = float(num_str)

                if key in key_list:
                    index = key_list.index(key)
                else:
                    key_list.append(key)
                    images.append([])
                    diopt.append([])
                    index = len(images) - 1  

                if blasen:
                    images[index].insert(0, cv2.imread(os.path.join(self.image_directory[0], filename), cv2.IMREAD_GRAYSCALE))
                    diopt[index].insert(0, num)
                else:
                    images[index].append(cv2.imread(os.path.join(self.image_directory[0], filename), cv2.IMREAD_GRAYSCALE))
                    diopt[index].append(num)

        return images, diopt
    
    def load_images_from_directory2(self):
        images = []  
        diopt = []   
        key_list = []  

        for filename in os.listdir(self.image_directory[1]):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                base_name, ext = os.path.splitext(filename)
                parts = base_name.split("_")

                blasen = parts[3] == 'after'

                key = int(parts[2][1])

                num_str = parts[-1].replace(",", ".")
                num = float(num_str)

                if key in key_list:
                    index = key_list.index(key)
                else:
                    key_list.append(key)
                    images.append([])
                    diopt.append([])
                    index = len(images) - 1  

                if blasen:
                    images[index].insert(0, cv2.imread(os.path.join(self.image_directory[1], filename), cv2.IMREAD_GRAYSCALE))
                    diopt[index].insert(0, num)
                else:
                    images[index].append(cv2.imread(os.path.join(self.image_directory[1], filename), cv2.IMREAD_GRAYSCALE))
                    diopt[index].append(num)

        return images, diopt
    
    def load_images_from_directory3(self):
        images = []  
        diopt = []   
        key_list = []  

        for filename in os.listdir(self.image_directory[2]):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                base_name, ext = os.path.splitext(filename)
                parts = base_name.split("_")

                blasen = parts[3] == 'after'

                key = int(parts[2][2])

                num_str = parts[-1].replace(",", ".")
                num = float(num_str)

                if key in key_list:
                    index = key_list.index(key)
                else:
                    key_list.append(key)
                    images.append([])
                    diopt.append([])
                    index = len(images) - 1  

                if blasen:
                    images[index].insert(0, cv2.imread(os.path.join(self.image_directory[2], filename), cv2.IMREAD_GRAYSCALE))
                    diopt[index].insert(0, num)
                else:
                    images[index].append(cv2.imread(os.path.join(self.image_directory[2], filename), cv2.IMREAD_GRAYSCALE))
                    diopt[index].append(num)

        return images, diopt
    
    def load_images_from_directory4(self):
        images = []  
        diopt = []   
        key_list = []  

        for filename in os.listdir(self.image_directory[3]):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                base_name, ext = os.path.splitext(filename)
                parts = base_name.split("_")

                blasen = parts[3] == 'after'

                key = int(parts[2][1])

                num_str = parts[-1].replace(",", ".")
                num = float(num_str)

                if key in key_list:
                    index = key_list.index(key)
                else:
                    key_list.append(key)
                    images.append([])
                    diopt.append([])
                    index = len(images) - 1  

                if blasen:
                    images[index].insert(0, cv2.imread(os.path.join(self.image_directory[3], filename), cv2.IMREAD_GRAYSCALE))
                    diopt[index].insert(0, num)
                else:
                    images[index].append(cv2.imread(os.path.join(self.image_directory[3], filename), cv2.IMREAD_GRAYSCALE))
                    diopt[index].append(num)

        return images, diopt
    
    def load_images_from_directory5(self):
        images = []  
        diopt = []   
        key_list = []  

        for filename in os.listdir(self.image_directory[4]):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                base_name, ext = os.path.splitext(filename)
                parts = base_name.split("_")

                blasen = parts[3] == 'after'

                key = int(parts[2][1])

                num_str = parts[-1].replace(",", ".")
                num = float(num_str)

                if key in key_list:
                    index = key_list.index(key)
                else:
                    key_list.append(key)
                    images.append([])
                    diopt.append([])
                    index = len(images) - 1  

                if blasen:
                    images[index].insert(0, cv2.imread(os.path.join(self.image_directory[4], filename), cv2.IMREAD_GRAYSCALE))
                    diopt[index].insert(0, num)
                else:
                    images[index].append(cv2.imread(os.path.join(self.image_directory[4], filename), cv2.IMREAD_GRAYSCALE))
                    diopt[index].append(num)

        return images, diopt
    
    def crop_images(self, x_offset, y_offset):
        cropped_images = []
        for img_ in self.images[0]:
            cropped_images_row = [] 
            for img in img_:
                if img.shape != (self.target_height, self.target_width):
                    original_height, original_width = img.shape[:2]
                    left = ((original_width - self.target_width) // 2 ) + x_offset
                    top = ((original_height - self.target_height) // 2 ) + y_offset
                    right = left + self.target_width 
                    bottom = top + self.target_height 

                    cropped_img = img[top:bottom, left:right]
                    cropped_images_row.append(cropped_img) 

                else:
                    cropped_images_row.append(img) 
                cropped_images.append(cropped_images_row) 
        self.images = cropped_images
        return cropped_images
    
class Merkmalsextraktion:
    def __init__(self, images):
        self.images = images

    # Sharpen Versuch 1
    def highpass_sharpen(self, kernel_size=3, alpha=1.0):
        highpassed_images = []
        for img_ in self.images:
            highpassed_images_rows = []
            for img in img_:
                laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=kernel_size, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

                highpassed_images_rows_uncliped = img + alpha * laplacian

                highpassed_images_rows.append(np.clip(highpassed_images_rows_uncliped, 0, 255).astype(np.uint8))
            highpassed_images.append(highpassed_images_rows)

        return highpassed_images
    
    # Sharpen Versuch 2
    def unsharp_mask(self, kernel_size=(11, 11), sigma=3, amount=4.0, threshold=10): # kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0
        sharpened_images = []
        for img_ in self.images:
            sharpened_images_row = [] 
            for img in img_:
                blurred = cv2.GaussianBlur(img, kernel_size, sigma)
                sharpened = int(amount + 1) * img - int(amount) * blurred # float was exchanged by an int
                sharpened_images_row.append(np.maximum(sharpened, np.zeros_like(img)))
            sharpened_images.append(sharpened_images_row)
        return sharpened_images

    # Sharpen Versuch 3
    def apply_canny(self):
        canny_images = [[cv2.Canny(img, 10, 300, L2gradient=True, apertureSize=7) for img in row] for row in self.images]
        return canny_images
    
    # Feature Extraction 
    def average_images(self):
        averaged_images = []
        for img_ in self.images:
            averaged_images_row = []
            for img in img_:
                averaged_images_row.append(np.mean(img))
            averaged_images.append(averaged_images_row)
        return averaged_images
    
    def process_inner_lists_images(self):
        first_elements = []
        second_elements = []
        unique_inner_lists = []
        cont = 0

        for inner_list in self.images:
            if len(inner_list) == 1:
                continue
            
            first_elem, second_elem = inner_list[0], inner_list[1]

            first_elements.append(first_elem)
            second_elements.append(second_elem)

            if cont != 0:
                # Check if element at index 0 matches any other element in the list
                if any((first_elem == other_elem).all() for other_elem in unique_inner_lists[:cont][0][0]):
                    continue  # Skip this inner list

                # Check if element at index 1 matches any other element in the list
                if any((second_elem == other_elem).all() for other_elem in unique_inner_lists[:cont][0][1]):
                    continue  # Skip this inner list

            unique_inner_lists.append(inner_list)
            cont += 1

        return first_elements, second_elements, unique_inner_lists
    
    def process_inner_lists(self):
        first_elements = []
        second_elements = []
        unique_inner_lists = []

        for inner_list in self.images:
            if len(inner_list) == 1:
                continue

            first_elem, second_elem = inner_list[0], inner_list[1]

            first_elements.append(first_elem)
            second_elements.append(second_elem)

            if inner_list not in unique_inner_lists:
                unique_inner_lists.append(inner_list)

        return first_elements, second_elements, unique_inner_lists
    
    def process_diopts(self, diopts):
        doubled_diopts = []
        diopt_change = []

        for inner_list in diopts:
            if len(inner_list) == 2:
                doubled_diopts.append(inner_list)
                diopt_change.append(inner_list[0] - inner_list[1])

        return doubled_diopts, diopt_change
    
    def t_test(self, first_elements, second_elements):
        blasen_list = first_elements
        blasenlos_list = second_elements

        t_statistic, p_value = stats.ttest_ind(blasen_list, blasenlos_list)
        alpha = 0.05

        if p_value < alpha:
            print(f"The p-value ({p_value:.4f}) is less than {alpha:.2f}. The lists are statistically significantly different.")
        else:
            print(f"The p-value ({p_value:.4f}) is greater than {alpha:.2f}. The lists are not statistically significantly different.")

    def detect_circles(self):
        circles_list = []
        circles_check = []
        for img_ in self.images:
            circles_list_rows = []
            circles_check_rows = []
            for img in img_:
                circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=500,
                                       param1=100, param2=100, minRadius=100, maxRadius=250)
                if circles is not None:
                    circles_list_rows.append(circles)
                    circles_check_rows.append(1)
                    print('Circle found!')
                else:
                    circles_list_rows.append(0)
                    circles_check_rows.append(0)
                print('Image completed')
            circles_list.append(circles_list_rows)
            circles_check.append(circles_check_rows)
        return circles_list, circles_check

    def extract_features(self, circles_list, circles_check):
        features = []
        cont = 0
        for circles in circles_list:
            if circles_check[cont] == 1:
                num_circles = circles.shape[1]
                centers = circles[:, :, :2]
                radii = circles[:, :, -1]           
                avg_intensities = []
                for center, radius in zip(centers[0], radii[0]):
                    x, y = int(center[0]), int(center[1])
                    circle_mask = np.zeros_like(self.images[cont])
                    cv2.circle(circle_mask, (x, y), int(radius), 1, -1)  
                    avg_intensity = np.mean(self.images[cont][circle_mask == 1])
                    avg_intensities.append(avg_intensity)
                
                features.append([num_circles, np.mean(circles[:, :, 0]), np.mean(circles[:, :, 1]), np.mean(radii), np.mean(avg_intensities)])
            else:
                features.append([0, 0, 0, 0, 0])
            cont += 1
        return features

    def normalize_features(self, features):
        scaler = MinMaxScaler()
        return scaler.fit_transform(features)
    
class RegressionModel:
    def __init__(self, input_shape):
        self.model = self.create_regression_model(input_shape)

    def create_regression_model(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)  
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_model(self, X, y, validation_data, epochs=100, batch_size=32):
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plotTrainingHistory(history):
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def get_weights_and_biases(self):
        weights, biases = self.model.layers[0].get_weights()
        return weights, biases
    
    def make_predictions_and_evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)

        plt.scatter(X_test, y_test, label='Data')
        plt.plot(X_test, y_pred, color='red', label='Predictions')
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.legend()
        plt.show()

        test_loss = model.evaluate(X_test, y_test)
        return test_loss

if __name__ == "__main__":
    image_directory = [r"C:\Users\SANCHDI2\OneDrive - Alcon\Desktop\Dioptrie1", 
                       r"C:\Users\SANCHDI2\OneDrive - Alcon\Desktop\Dioptrie2", 
                       r"C:\Users\SANCHDI2\OneDrive - Alcon\Desktop\Dioptrie3", 
                       r"C:\Users\SANCHDI2\OneDrive - Alcon\Desktop\Dioptrie4",
                       r"C:\Users\SANCHDI2\OneDrive - Alcon\Desktop\Dioptrie5"
                       ]  
    image_processor = Bildvorverarbeitung(image_directory)

    images = image_processor.images[0]
    diopts = image_processor.images[1]

    resized_images = image_processor.crop_images(x_offset=-100, y_offset=1650) # Mit dieser Implementierung wird jeder Bild gecropt und geoffset
                                                                               # x_offset=-100, y_offset=1650
    x_dim_img = 750 # 650 (sehr zentriert) # 4504 (ganz Bild)
    y_dim_img = 750 # 650 (sehr zentriert)
    z_dim_img = 1

    sharpen_image = Merkmalsextraktion(resized_images) 
    sharpened_images = sharpen_image.unsharp_mask()

    #highpass_image = Merkmalsextraktion(resized_images)
    #highpassed_images = highpass_image.highpass_sharpen()

    #canny_edged_image = Merkmalsextraktion(resized_images)
    #canny_edged_images = canny_edged_image.apply_canny()

    input_images = sharpened_images

    process_images = Merkmalsextraktion(input_images)
    first_images, second_images, unique_inner_images = process_images.process_inner_lists_images()
    first_images = [ele for idx, ele in enumerate(first_images) if idx % 2 == 0]
    second_images = [ele for idx, ele in enumerate(second_images) if idx % 2 == 0]
    unique_inner_images = [ele for idx, ele in enumerate(unique_inner_images) if idx % 2 == 0]

    process_diopt = Merkmalsextraktion(input_images)
    diopts, diopt_change = process_diopt.process_diopts(diopts)

    ##################################################################################################################################################################################
    
    # Convert original labels to mapped labels
    y = diopt_change

    # Split data into training, validation, and test sets
    X = first_images

    # Split data into train and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Normalize pixel values to [0, 1]
    for i in range(len(X_train)):
        X_train[i] = np.divide(X_train[i], 255)
    for i in range(len(X_val)):
        X_val[i] = np.divide(X_val[i], 255)
    for i in range(len(X_test)):
        X_test[i] = np.divide(X_test[i], 255)

    X_train = np.array(X_train)
    X_val   = np.array(X_val)
    X_test  = np.array(X_test)

    # Check which types of images are you inputting
    image_plotter = BildPlotter(X_train[0]) 
    image_plotter.plot_image()
    plt.title('Sharpened Example')

    y_train = np.array(y_train)
    y_val   = np.array(y_val)
    y_test  = np.array(y_test)

    # Assuming X_train, X_val, y_train, y_val are your data arrays
    X_train_rgb = np.repeat(X_train[..., np.newaxis], 3, axis=-1)
    X_val_rgb = np.repeat(X_val[..., np.newaxis], 3, axis=-1)
    X_test_rgb = np.repeat(X_test[..., np.newaxis], 3, axis=-1)

    ############################################################################################################################################################################

    # Load the pre-trained VGG16 model (excluding the top classification layers)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(750, 750, 3)) 

    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers for regression
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    output_layer = Dense(1, activation='linear')(x)  # Regression output

    # Create the final regression model
    model = Model(inputs=base_model.input, outputs=output_layer)

    # Compile the model with mean squared error (MSE) loss
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Summary of the modified model architecture
    model.summary()

    # Train the model
    history = model.fit(X_train_rgb, y_train, epochs=20, batch_size=16, validation_data=(X_val_rgb, y_val)) # To feed

    test_loss, test_mae = model.evaluate(X_test_rgb, y_test)
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')  
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.show()

    y_pred_test = model.predict(X_test_rgb)

    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='r', linestyle='--', label='Ideal Line')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs. True Values')
    plt.show()

    print('Hello World')