import os
import cv2
import pywt
import keras
from keras import layers
import numpy as np
import pandas as pd
import openpyxl
import tensorflow as tf
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from decimal import Decimal

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
    def __init__(self, image_directory, excel_directory, target_height, target_width): # target_height=4504 target_width=4504
        self.image_directory = image_directory
        self.excel_directory = excel_directory                                         # target_height=750, target_width=750
        self.target_height   = target_height                                    
        self.target_width    = target_width
        self.images          = self.load_images_from_directory_call()

    def load_images_from_directory_call(self):
        images, diopt = self.load_images_from_directory()
        return images, diopt
    
    def load_images_from_directory(self):
        images = []  
        diopt = []   
        key_list = []  

        df = pd.read_excel(self.excel_directory) 
        title_exc = df['Title'].tolist()
        sample_exc = df['Sample'].tolist()

        complete_directory = []
        for directories in self.image_directory:
            complete_directory.extend(os.listdir(directories))

        for filename in complete_directory:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                base_name, ext = os.path.splitext(filename)
                parts = base_name.split("_")

                title_img = parts[2][:6]
                sample_img = parts[2][7:]

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
    
if __name__ == "__main__":
    image_directory = [r"C:\Users\SANCHDI2\OneDrive - Alcon\Desktop\Labeled1", 
                       r"C:\Users\SANCHDI2\OneDrive - Alcon\Desktop\Labeled2", 
                       ]  
    excel_directory = "example.xlsx"
    image_processor = Bildvorverarbeitung(image_directory, excel_directory, target_height=750, target_width=750)

    images = image_processor.images[0]
    diopts = image_processor.images[1]