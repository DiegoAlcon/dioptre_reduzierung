import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Erstens: Bildvorverarbeitung
# Aufgabe:
# - Läd ein echtes Bild hoch
# - Ändert die Größe des Bildes zu einem bestimmte Größe
# - Wandelt das Bild zu einem Grayskala Bild um
# - Erkenn die Grenze des Bildes mit Canny Edge Erkennung 

# Zweitens: Merkamlsextraktion
# Aufgabe:
# - Erkenn die Blasen mit der Hough Transformation
# - Extrahier die vorliegende Merkmale:
#   - Anzahl der Kreise (Blasen).
#   - Durchschnitt der X-Koordinaten aller Kreise.
#   - Durchschnitt der Y-Koordinaten aller Kreis.
#   - Durchschnitt des Durchmessens aller Kreise.
#   - Durchschnitt der Intensität der Pixel aller Kreise.
# Normaliziert alle die Dateien 

# Drittens: Dioptrieänderung Bestimmung
# Aufgabe:
# Stell ein Regressionsalgorithmus mit dem Merkmalsvektor als Eingang (5 inputs/Bild) und der Dioptrieänderung als Ausgang (1 output/Bild).
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
    def __init__(self, image_directory, target_height=550, target_width=550): # target_height=4504 target_width=4504
        self.image_directory = image_directory                                 # target_height=200, target_width=1300
        self.target_height   = target_height
        self.target_width    = target_width
        self.images          = self.load_images_from_directory()
    
    def load_images_from_directory(self):
        images = []  # List of lists to store images
        diopt = []   # List of lists to store diopt values
        key_list = []  # List to keep track of unique keys

        for filename in os.listdir(self.image_directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Extract information from the filename
                base_name, ext = os.path.splitext(filename)
                parts = base_name.split("_")

                # Determine if it's the "first position" or "second position"
                blasen = parts[1] == 'after'

                # Extract the key (integer) from the filename
                key = int(parts[0][9:11])

                # Extract the last 5 characters (replace "," with ".")
                num_str = parts[-1].replace(",", ".")
                num = float(num_str)

                # Check if the key is already in the key_list
                if key in key_list:
                    # Find the index of the key in the key_list
                    index = key_list.index(key)
                else:
                    # Add the key to the key_list and create new lists for images and diopt
                    key_list.append(key)
                    images.append([])
                    diopt.append([])
                    index = len(images) - 1  # Index of the newly added lists

                # Store image data based on the blasen value
                if blasen:
                    images[index].insert(0, cv2.imread(os.path.join(self.image_directory, filename), cv2.IMREAD_GRAYSCALE))
                    diopt[index].insert(0, num)
                else:
                    images[index].append(cv2.imread(os.path.join(self.image_directory, filename), cv2.IMREAD_GRAYSCALE))
                    diopt[index].append(num)

        return images, diopt

    
    # Sort images based on the key
        #sorted_keys = sorted(images.keys())
        #sorted_images = [images[key] for key in sorted_keys]
        #sorted_diopt = [diopt[key] for key in sorted_keys]
    
    #def check_image_sizes(self):
    #    first_image_height, first_image_width = self.images[0].shape
    #    for img in self.images[1:]:
    #        if img.shape != (first_image_height, first_image_width):
    #            return False
    #    return True
    
    # Change the code so that always the steps are done and crop (if necessary and offset are performed)
    def crop_images(self, x_offset, y_offset):
        cropped_images = []
        for img_ in self.images[0]:
            cropped_images_row = [] #
            for img in img_:
                if img.shape != (self.target_height, self.target_width):
                    original_height, original_width = img.shape[:2]
                    left = ((original_width - self.target_width) // 2 ) + x_offset
                    top = ((original_height - self.target_height) // 2 ) + y_offset
                    right = left + self.target_width 
                    bottom = top + self.target_height 

                    cropped_img = img[top:bottom, left:right]
                    cropped_images_row.append(cropped_img) #

                else:
                    cropped_images_row.append(img) #
                cropped_images.append(cropped_images_row) #
        self.images = cropped_images
        return cropped_images
    
class Merkmalsextraktion:
    def __init__(self, images):
        self.images = images

    # Sharpen Versuch 3
    def highpass_sharpen(self, kernel_size=3, alpha=1.0):
        # Convert the image to grayscale if it's color
        #if len(image.shape) == 3:
        #    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        highpassed_images = []
        for img_ in self.images:
            highpassed_images_rows = []
            for img in img_:
                # Apply Laplacian filter to compute the highpass component
                laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=kernel_size, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

                # Combine the original image with the highpass component
                highpassed_images_rows_uncliped = img + alpha * laplacian

                # Clip pixel values to [0, 255]
                highpassed_images_rows.append(np.clip(highpassed_images_rows_uncliped, 0, 255).astype(np.uint8))
            highpassed_images.append(highpassed_images_rows)

        return highpassed_images
    
    # Sharpen Versuch 2
    def unsharp_mask(self, kernel_size=(11, 11), sigma=3, amount=4.0, threshold=10): # kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0
        sharpened_images = []
        for img_ in self.images:
            sharpened_images_row = [] #
            for img in img_:
                blurred = cv2.GaussianBlur(img, kernel_size, sigma)
                sharpened = int(amount + 1) * img - int(amount) * blurred # float was exchanged by an int
                sharpened_images_row.append(np.maximum(sharpened, np.zeros_like(img)))
            sharpened_images.append(sharpened_images_row)
        return sharpened_images

    # Sharpen Versuch 3
    def apply_canny(self):
        #return [cv2.Canny(img, 100, 200) for img in self.images]
        canny_images = [[cv2.Canny(img, 10, 300, L2gradient=True, apertureSize=7) for img in row] for row in self.images]
        return canny_images

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
                    #circles_list.append(circles)
                    #circles_check.append(1)
                    circles_list_rows.append(circles)
                    circles_check_rows.append(1)
                    print('Circle found!')
                else:
                    #circles_list.append(0)
                    #circles_check.append(0)
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
        #self.model.fit(X, y, epochs=epochs, batch_size=batch_size) 
        # Train the model
        # return self.model.fit(X, y, epochs=epochs, batch_size=batch_size)
        return self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

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

if __name__ == "__main__":
    image_directory = r"C:\Users\SANCHDI2\OneDrive - Alcon\Desktop\Dioptrie"  
    image_processor = Bildvorverarbeitung(image_directory)

    images = image_processor.images[0]
    diopts = image_processor.images[1]

    # image_processor.images[0] returns a list of 70 lists each of 2 images
    #    images with bubbles are in the first position of each inner list, each can be accessed as: image_processor.images[0][n][0] where n € 0:69 
    #    images without bubbles are in the second position of each inner list, each can be accessed as: image_processor.images[0][n][1] where n € 0:69 
    # Dioptre factors have the same behaviour, each correspond to one image and are stored in: image_processor.images[1]

    image_plotter = BildPlotter(image_processor.images[0][0][0]) 
    image_plotter.plot_image()

    #if not image_processor.check_image_sizes():
    #    resized_images = image_processor.crop_images()
    #    print(f"All images resized to {image_processor.target_height}x{image_processor.target_width} pixels.")
    #else:
    #    resized_images = image_processor.images
    #    print("All images already have the same size.")

    resized_images = image_processor.crop_images(x_offset=-100, y_offset=1650) # Mit dieser Implementierung wird jeder Bild gecropt und geoffset
                                                                               # x_offset=-100, y_offset=1000

    image_plotter = BildPlotter(resized_images[0][0]) 
    image_plotter.plot_image()

    resized_images = resized_images[:1] # If active, the following three (*) might not be important

    #highpass_image = Merkmalsextraktion(resized_images) # To apply highpass
    #highpassed_images = highpass_image.highpass_sharpen()

    #image_plotter = BildPlotter(highpassed_images[0][0]) # To plot highpass
    #image_plotter.plot_image()

    #sharpen_image = Merkmalsextraktion(resized_images) # To apply kernel
    #sharpened_images = sharpen_image.unsharp_mask()

    #image_plotter = BildPlotter(sharpened_images[0][0]) # To plot kernel
    #image_plotter.plot_image()

    #feature_extractor = Merkmalsextraktion(resized_images) # Apply canny
    #canny_images = feature_extractor.apply_canny()
    
    #image_plotter = BildPlotter(canny_images[0][0]) # To plot canny
    #image_plotter.plot_image()

    #canny_images = canny_images[:1] # to be removed # This three might not be important (*)
    #resized_images = resized_images[:1] # to be removed
    #sharpened_images = sharpened_images[:1] # to be removed

    #feature_extractor = Merkmalsextraktion(canny_images) # One of the following five must be active
    #feature_extractor = Merkmalsextraktion(highpassed_images)
    #feature_extractor = Merkmalsextraktion(sharpened_images)
    feature_extractor = Merkmalsextraktion(resized_images)
    #feature_extractor = Merkmalsextraktion(images[0])
    
    circles_list, circles_check = feature_extractor.detect_circles() # This must be active

    image_plotter = BildCirclesPlotter(images[0][0], circles_list[0][0]) # To plot
    image_plotter.plot_circles()
    image_plotter = BildCirclesPlotter(images[0][1], circles_list[0][1]) 
    image_plotter.plot_circles()