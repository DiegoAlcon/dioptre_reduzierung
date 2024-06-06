import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

class ImageProcessor:
    def __init__(self, image_directory, target_height=128, target_width=128):
        self.image_directory = image_directory
        self.target_height = target_height
        self.target_width = target_width
        self.images = self.load_images_from_directory()

    def load_images_from_directory(self):
        images = []
        for filename in os.listdir(self.image_directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img = cv2.imread(os.path.join(self.image_directory, filename), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
        return images

    def check_image_sizes(self):
        first_image_height, first_image_width = self.images[0].shape
        for img in self.images[1:]:
            if img.shape != (first_image_height, first_image_width):
                return False
        return True

    def resize_images(self):
        resized_images = []
        for img in self.images:
            if img.shape != (self.target_height, self.target_width):
                resized_img = cv2.resize(img, (self.target_width, self.target_height))
                resized_images.append(resized_img)
            else:
                resized_images.append(img)
        return resized_images

class FeatureExtractor:
    def __init__(self, images):
        self.images = images

    def apply_canny(self):
        return [cv2.Canny(img, 100, 200) for img in self.images]

    def detect_circles(self):
        circles_list = []
        for img in self.images:
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                       param1=50, param2=30, minRadius=5, maxRadius=50)
            if circles is not None:
                circles_list.append(circles)
        return circles_list

    def extract_features(self, circles_list):
        features = []
        for circles in circles_list:
            num_circles = len(circles)
            centers = circles[:, :2]
            radii = circles[:, 2]
            avg_intensity = np.mean([img[int(center[1]), int(center[0])] for img, center in zip(self.images, centers)]) # Error while converting int(center[1]) to a python scalar
            features.append([num_circles, np.mean(centers, axis=0), np.mean(radii), avg_intensity])
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

    def train_model(self, X, y, epochs=100, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def get_weights_and_biases(self):
        weights, biases = self.model.layers[0].get_weights()
        return weights, biases

if __name__ == "__main__":
    image_directory = r"C:\Users\SANCHDI2\OneDrive - Alcon\Desktop\Circle Images"  # Replace with the actual path
    image_processor = ImageProcessor(image_directory)

    if not image_processor.check_image_sizes():
        resized_images = image_processor.resize_images()
        print(f"All images resized to {image_processor.target_height}x{image_processor.target_width} pixels.")
    else:
        resized_images = image_processor.images
        print("All images already have the same size.")

    feature_extractor = FeatureExtractor(resized_images)
    canny_images = feature_extractor.apply_canny()
    circles_list = feature_extractor.detect_circles()
    features = feature_extractor.extract_features(circles_list)
    normalized_features = feature_extractor.normalize_features(features)

    input_shape = (normalized_features.shape[1],)
    regression_model = RegressionModel(input_shape)

    y = np.array([your_continuous_output_value] * len(normalized_features)) # To be replaced later
    regression_model.train_model(normalized_features, y)

    weights, biases = regression_model.get_weights_and_biases()
    expression = f"y = {weights[0][0]} * x1 + {weights[1][0]} * x2 + {weights[2][0]} * x3 + {biases[0]}"
    print("Regression model trained successfully.")
    print("Mathematical expression:", expression)
