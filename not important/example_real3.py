import os
import cv2
import pywt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from decimal import Decimal

# Erstens: Bildvorverarbeitung
# Aufgabe:
# - Läd ein echtes Bild hoch
# - Ändert die Größe des Bildes zu einem bestimmte Größe
# - Wandelt das Bild zu einem Grayskala Bild um
# - Offset die Position des Bildes um auf die Blasen zu fokusieren 

# Zweitens: Merkamlsextraktion
# Aufgabe:
# - Erkenn die Blasen mit der Reflektion des Blasens (d.h. der höher des Wertes der Pixels, desto hochmöglich, dass es ein Blase gibt)
# - Extrahier die vorliegende Merkmale:
#   - Anzahl der Kreise (Blasen). (Immer 1 = nicht wichtig)
#   - Durchschnitt der X-Koordinaten aller Kreise. (Nein)
#   - Durchschnitt der Y-Koordinaten aller Kreis. (Nein)
#   - Durchschnitt des Durchmessens aller Kreise. (Nein)
#   - Durchschnitt der Intensität der Pixel aller Kreise. (Ja)
# Normaliziert alle die Dateien 

# Drittens: Dioptrieänderung Bestimmung
# Aufgabe:
# Stell ein Regressionsalgorithmus mit dem Merkmalsvektor als Eingang (1 inputs/Bild) und der Dioptrieänderung als Ausgang (1 output/Bild).
# Oder:
# Stell ein Regressionsalgorithmus mit dem Merkmalsvektor als Eingang (N-Pixelwerte inputs/Bild) und der Dioptrieänderung als Ausgang (1 output/Bild).
# Trainiert dieser Algorithmus mit 80% der ganze Datenbank (nur mit Blasen) zufälling. 
#   - Bilder ohne Blasen können auch als Eingang angenommen werden, soweit die Dioptrieänderung auf Null gesetzt sei. 
# Validiert dieser Algorithmus mit den Rest 20% der ganze Datenbank (nur mit Blasen). 

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
    def __init__(self, image_directory, target_height=650, target_width=650): # target_height=4504 target_width=4504
        self.image_directory = image_directory                                 # target_height=200, target_width=1300
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
    
    # Feature Extraction #################################################################################################################################
    def average_images(self):
        averaged_images = []
        for img_ in self.images:
            averaged_images_row = []
            for img in img_:

                #coeffs2 = pywt.dwt2(img, 'bior1.3')
                #LL, (LH, HL, HH) = coeffs2
                #energy = (LH**2 + HL**2 + HH**2).sum() / img.size
                #averaged_images_row.append(energy)

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

    # image_processor.images[0] returns a list of 70 lists each of 2 images
    #    images with bubbles are in the first position of each inner list, each can be accessed as: image_processor.images[0][n][0] where n € 0:69 
    #    images without bubbles are in the second position of each inner list, each can be accessed as: image_processor.images[0][n][1] where n € 0:69 
    # Dioptre factors have the same behaviour, each correspond to one image and are stored in: image_processor.images[1]

    #image_plotter = BildPlotter(image_processor.images[0][0][0]) 
    #image_plotter.plot_image()

    resized_images = image_processor.crop_images(x_offset=-100, y_offset=1650) # Mit dieser Implementierung wird jeder Bild gecropt und geoffset
                                                                               # x_offset=-100, y_offset=1000
    #image_plotter = BildPlotter(resized_images[0][0]) 
    #image_plotter.plot_image()

    #sharpen_image = Merkmalsextraktion(resized_images) 
    #sharpened_images = sharpen_image.unsharp_mask()

    #highpass_image = Merkmalsextraktion(resized_images)
    #highpassed_images = highpass_image.highpass_sharpen()

    #canny_edged_image = Merkmalsextraktion(resized_images)
    #canny_edged_images = canny_edged_image.apply_canny()

    input_images = resized_images

    process_images = Merkmalsextraktion(input_images)
    first_images, second_images, unique_inner_images = process_images.process_inner_lists_images()
    first_images = [ele for idx, ele in enumerate(first_images) if idx % 2 == 0]
    second_images = [ele for idx, ele in enumerate(second_images) if idx % 2 == 0]
    unique_inner_images = [ele for idx, ele in enumerate(unique_inner_images) if idx % 2 == 0]

    process_diopt = Merkmalsextraktion(input_images)
    diopts, diopt_change = process_diopt.process_diopts(diopts)

    # Sample the label domain
    num_labels = 11
    #label_range = np.arange(-5, 6)  # Create an array of labels
    #label_range = np.round(label_range, 1) # needed if arange specifies a third argument with 0.1
    #label_range = np.round(label_range)
    label_range = np.array([0,  1,  2,  3,  4,  5, -5, -4, -3, -2, -1])
    #label_mapping = {label: i for i, label in enumerate(label_range)}
    #inverse_label_mapping = {v: k for k, v in label_mapping.items()}

    # Convert original labels to mapped labels
    y = diopt_change
    #y = [round(elem, 1) for elem in y]
    y = [round(elem) for elem in y] 

    ####################################################################################################################################################################

    # Step 1: Create a histogram for specific decimal values
    #plt.figure(figsize=(8, 6))
    #plt.hist(y, bins=np.arange(min(y), max(y) + 0.1, 0.1), alpha=0.7, label='Specific Decimal Values')
    #plt.xlabel('Decimal Values')
    #plt.ylabel('Frequency')
    #plt.title('Histogram for Specific Decimal Values')
    #plt.legend()
    #plt.grid(True)
    #plt.show()

    # Step 2: Create a histogram for integer values
    #integer_values = [int(Decimal(str(val))) for val in y]
    #plt.figure(figsize=(8, 6))
    #plt.hist(integer_values, bins=np.arange(min(integer_values), max(integer_values) + 1, 1), alpha=0.7, color='orange', label='Integer Values')
    #plt.xlabel('Integer Values')
    #plt.ylabel('Frequency')
    #plt.title('Histogram for Integer Values')
    #plt.legend()
    #plt.grid(True)
    #plt.show()

    #######################################################################################################################################################################

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
    X_test = np.array(X_test)
    X_val = np.array(X_val)

    ################################################################################################
    # Check which types of images are you inputting
    #image_plotter = BildPlotter(X_train[0]) 
    #image_plotter.plot_image()
    #plt.title('Sharpened Example')
    #plt.show()

    ################################################################################################

    # Define the CNN architecture
    #model = models.Sequential()
    #model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(550, 550, 1)))
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Flatten())
    #model.add(layers.Dense(128, activation='relu'))
    #model.add(layers.Dense(num_labels, activation='softmax'))  # Output layer with num_labels neurons
    
    #one-hot encode target column
    #y_train = np.array(y_train)*10
    #y_test = np.array(y_test)*10
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_val  = np.array(y_val)
    #condition1 = y_train > 49
    #condition2 = y_train < -49
    #y_train[condition1] = 49
    #y_train[condition2] = -49
    #condition1 = y_test > 49
    #condition2 = y_test < -49
    #y_test[condition1] = 49
    #y_test[condition2] = -49
    condition1 = y_train > 5
    condition2 = y_train < -5
    y_train[condition1] = 5
    y_train[condition2] = -5
    condition1 = y_test > 5
    condition2 = y_test < -5
    y_test[condition1] = 5
    y_test[condition2] = -5
    condition1 = y_val > 5
    condition2 = y_val < -5
    y_val[condition1] = 5
    y_val[condition2] = -5
    y_train_not_categorical = y_train
    y_test_not_categorical = y_test
    y_val_not_categorical = y_val
    y_train = to_categorical(y_train, num_classes=num_labels)
    y_test = to_categorical(y_test, num_classes=num_labels)
    y_val = to_categorical(y_val, num_classes=num_labels)
    #y_train = np.array(y_train)
    #y_test = np.array(y_test)
    print(f"One-hot vector example: {y_train[0]}")

    #create model
    #model = Sequential()
    #add model layers
    x_dim_img = 650
    y_dim_img = 650
    z_dim_img = 1
    #model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(x_dim_img, y_dim_img, z_dim_img)))
    #model.add(Conv2D(32, kernel_size=3, activation='relu'))
    #model.add(Flatten())
    #model.add(Dense(num_labels, activation='softmax')) # Softmax is used for CNN

    # Define the CNN aexample_real3.pyrchitecture (for CNN CATEGORICAL)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(x_dim_img, y_dim_img, z_dim_img)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_labels, activation='softmax'))  # Output layer with num_labels neurons

    # Define the CNN architecture (for CNN REGRESSION)
    #model = models.Sequential()
    #model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(x_dim_img, y_dim_img, z_dim_img)))
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Flatten())
    #model.add(layers.Dense(128, activation='relu'))
    #model.add(layers.Dense(1, activation='tanh'))  # Single output neuron with tanh activation

    #train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    #train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size=64)

    #val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    #val_dataset = val_dataset.batch(batch_size=64)

    #test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    #test_dataset = test_dataset.batch(batch_size=64)

    #y_train_one_hot = (y_train, num_labels)
    #y_val_one_hot = (y_val, num_labels)
    #y_test_one_hot = (y_test, num_labels)

    # Compile the model
    #optimizer = Adam(learning_rate=0.1) # without this: no 0.001
    #model.compile(optimizer='adam', # just write: 'adam'
                  #loss='categorical_crossentropy',  # Use categorical crossentropy for integer labels
                 # metrics=['accuracy'])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])  # Use mean squared error loss for regression

    #X_train = tf.stack(X_train)
    #y_train = tf.stack(y_train)
    #X_val = tf.stack(X_val)
    #y_val = tf.stack(y_val)
    #X_test = tf.stack(X_test)
    #y_test = tf.stack(y_test)

    # Train the model
    #history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    #history = model.fit(X_train, y_train_one_hot, epochs=10, validation_data=(X_val, y_val_one_hot))
    #history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)
    #history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
    history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val))

    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

    # Plot the training and validation loss curves
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.show()

    # Evaluate the model on the test set
    #test_loss, test_accuracy = model.evaluate(X_test, y_test_one_hot)
    #test_loss, test_accuracy = model.evaluate(test_dataset)
    #print(f"Test accuracy: {test_accuracy:.4f}")

    # Make predictions
    predictions = model.predict(X_test)

    # Convert predictions back to original labels
    predicted_labels = [label_range[np.argmax(pred)] for pred in predictions]

    print(f"Here are the predictions: {predictions}")

    #actual results for first 4 images in test set
    print(f"Here are the actual values: {y_test}")

    #original_labels = [inverse_label_mapping[label] for label in y_test]
    #original_labels = {tensor.ref(): label for tensor, label in zip(y_test, inverse_label_mapping)}

    #plt.scatter(original_labels, predicted_labels, color='b', label='Predictions')
    plt.scatter(y_test_not_categorical, predicted_labels, color='b', label='Predictions')

    # Add the ideal line (45-degree line)
    plt.plot([min(y_test_not_categorical), max(y_test_not_categorical)], [min(y_test_not_categorical), max(y_test_not_categorical)], color='r', linestyle='--', label='Ideal Line')

    # Customize the plot
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title('Predicted vs. Ground Truth Scatter Plot')
    plt.legend()

    # Show the plot
    plt.show()
    print('Hello world')
    ######################################################################################################################################################

    #process_images = Merkmalsextraktion(unique_inner_images)
    #first_images, second_images, unique_inner_images = process_images.process_inner_lists_images()

    #average_image = Merkmalsextraktion(sharpened_images)
    #averaged_images = average_image.average_images()

    #process_list = Merkmalsextraktion(averaged_images)
    #first_elements, second_elements, unique_inner_lists = process_list.process_inner_lists()
    #process_list = Merkmalsextraktion(unique_inner_lists)
    #first_elements, second_elements, unique_inner_lists = process_list.process_inner_lists()

    #t_test = Merkmalsextraktion(averaged_images)
    #t_test.t_test(first_elements, second_elements)

    #feature_extractor = Merkmalsextraktion(canny_images) # One of the following five must be active
    #feature_extractor = Merkmalsextraktion(highpassed_images)
    #feature_extractor = Merkmalsextraktion(sharpened_images)
    #feature_extractor = Merkmalsextraktion(resized_images)
    #feature_extractor = Merkmalsextraktion(images[0])
    
    #circles_list, circles_check = feature_extractor.detect_circles() # This must be active

    #image_plotter = BildCirclesPlotter(images[0][0], circles_list[0][0]) # To plot
    #image_plotter.plot_circles()
    #image_plotter = BildCirclesPlotter(images[0][1], circles_list[0][1]) 
    #image_plotter.plot_circles()

    #x = np.array(first_elements) # Input
    #x_train = x[:50]
    #x_val = x[50:60]
    #x_test = x[60:]
    #regression_model = RegressionModel(x.shape) 
    #regression_model = RegressionModel((1,)) 

    #y = np.array(diopt_change) # output
    #y_train = y[:50]
    #y_val = y[50:60]
    #y_test = y[60:]

    #regression_model.train_model(x_train, y_train, validation_data=(x_val, y_val))

    #weights, biases = regression_model.get_weights_and_biases()
    #expression = f"y = {weights[0][0]} * x1 + {weights[1][0]} * x2 + {weights[2][0]} * x3 + {biases[0]}"
    #print("Regression model trained successfully.")
    #print("Mathematical expression:", expression)

    # Make predictions
    #y_pred = model.predict(x_test)
    #test_loss = regression_model.make_predictions_and_evaluate(regression_model.model, x_test, y_test)
    #print(f"Test loss: {test_loss}")

    # Plot mögliche data den Model zu füttern

    #x_bubbles_feature = x
    #x_no_bubbles_feature = np.array(second_elements)

    #plt.scatter(x_bubbles_feature, y, label='Data Points', color='b', marker='o')
    #plt.xlabel('Pixelwertdurchschnitt')
    #plt.ylabel('Drioptrieänderung')
    #plt.title('Images with bubbles')
    #plt.show()

    #plt.scatter(x_no_bubbles_feature, y, label='Data Points', color='b', marker='o')
    #plt.xlabel('Pixelwertdurchschnitt')
    #plt.ylabel('Drioptrieänderung')
    #plt.title('Images without bubbles')
    #plt.show()

    #x_base = np.linspace(0,1,num=72)
    #plt.scatter(x_base, x_bubbles_feature, label='With bubbles', color='b', marker='o')
    #plt.scatter(x_base, x_no_bubbles_feature, label='Without bubbles', color='g', marker='s')

    #plt.xlabel('Groups')
    #plt.ylabel('Pixelwertdurchschnitt')
    #plt.title('Comparison of Two Groups')
    #plt.legend()

    #plt.show()

    # Plot histogramns
    #plt.hist(sharpened_images[0][0].ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    #plt.hist(sharpened_images[0][0].ravel())
    #plt.title("Histogram of Grayscale Image")
    #plt.xlabel("Pixel Intensity")
    #plt.ylabel("Frequency")
    #plt.show()
    #plt.hist(sharpened_images[0][1].ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    #plt.hist(sharpened_images[0][1].ravel())
    #plt.title("Histogram of Grayscale Image")
    #plt.xlabel("Pixel Intensity")
    #plt.ylabel("Frequency")
    #plt.show()