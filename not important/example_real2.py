import os
import cv2
import pywt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

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
    def __init__(self, image_directory, target_height=550, target_width=550): # target_height=4504 target_width=4504
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

    image_plotter = BildPlotter(image_processor.images[0][0][0]) 
    image_plotter.plot_image()

    resized_images = image_processor.crop_images(x_offset=-100, y_offset=1650) # Mit dieser Implementierung wird jeder Bild gecropt und geoffset
                                                                               # x_offset=-100, y_offset=1000
    image_plotter = BildPlotter(resized_images[0][0]) 
    image_plotter.plot_image()

    sharpen_image = Merkmalsextraktion(resized_images) 
    sharpened_images = sharpen_image.unsharp_mask()

    average_image = Merkmalsextraktion(sharpened_images)
    averaged_images = average_image.average_images()

    process_list = Merkmalsextraktion(averaged_images)
    first_elements, second_elements, unique_inner_lists = process_list.process_inner_lists()
    process_list = Merkmalsextraktion(unique_inner_lists)
    first_elements, second_elements, unique_inner_lists = process_list.process_inner_lists()

    t_test = Merkmalsextraktion(averaged_images)
    t_test.t_test(first_elements, second_elements)

    process_diopt = Merkmalsextraktion(averaged_images)
    diopts, diopt_change = process_diopt.process_diopts(diopts)

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

    x = np.array(first_elements) # Input
    x_train = x[:50]
    x_val = x[50:60]
    x_test = x[60:]
    #regression_model = RegressionModel(x.shape) 
    regression_model = RegressionModel((1,)) 

    y = np.array(diopt_change) # output
    y_train = y[:50]
    y_val = y[50:60]
    y_test = y[60:]

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

    x_bubbles_feature = x
    x_no_bubbles_feature = np.array(second_elements)

    plt.scatter(x_bubbles_feature, y, label='Data Points', color='b', marker='o')
    plt.xlabel('Pixelwertdurchschnitt')
    plt.ylabel('Drioptrieänderung')
    plt.title('Images with bubbles')
    plt.show()

    plt.scatter(x_no_bubbles_feature, y, label='Data Points', color='b', marker='o')
    plt.xlabel('Pixelwertdurchschnitt')
    plt.ylabel('Drioptrieänderung')
    plt.title('Images without bubbles')
    plt.show()

    x_base = np.linspace(0,1,num=72)
    plt.scatter(x_base, x_bubbles_feature, label='With bubbles', color='b', marker='o')
    plt.scatter(x_base, x_no_bubbles_feature, label='Without bubbles', color='g', marker='s')

    plt.xlabel('Groups')
    plt.ylabel('Pixelwertdurchschnitt')
    plt.title('Comparison of Two Groups')
    plt.legend()

    plt.show()

    # Plot histogramns
    #plt.hist(sharpened_images[0][0].ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    plt.hist(sharpened_images[0][0].ravel())
    plt.title("Histogram of Grayscale Image")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()
    #plt.hist(sharpened_images[0][1].ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    plt.hist(sharpened_images[0][1].ravel())
    plt.title("Histogram of Grayscale Image")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()