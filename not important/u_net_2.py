import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras import layers
from scipy.stats import shapiro
import scipy.stats as stats

##############################################################################################################################################################################
# THIS PROGRAM ASSUMES THAT THE ONLY VARIABLE ON WHOSE OUTPUT THE DIOPTRIEÄNDERUNG DEPENDS IS THE BUBBLE SIZE
# IN THIS PROGRAM WE ASSUME THAT THE BUBBLE SIZE CAN BE APPROXIMATED AS THE NUMBER OF PIXELS THE BUBBLE COVERS IN THE IMAGE
##############################################################################################################################################################################

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
    
class Stats:
    def normal_distribution(self, data):
        _, p_value = shapiro(data)
        if p_value > 0.05:
            return True
        else:
            return False
     
    def variance(self, data):
        return np.var(data)
    
    def t_test(self, data1, data2, equal_variance):
        t_statistic, p_value = stats.ttest_ind(data1, data2, equal_var=equal_variance)
        return p_value

if __name__ == "__main__":
    # Klein Rechner
    image_directory = [r"C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\Labeled1", 
                       r"C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\Labeled2", 
                       ]  
    excel_directory = "example.xlsx"
    image_processor = Bildvorverarbeitung(image_directory, excel_directory, target_height=900, target_width=900, x_offset=-225, y_offset=1250)

    diopts = image_processor.images[1]

    del diopts[55] # Make sure that the program binary_mask_creation_2 has the following line: del images[55] --> otherwise: error

    original_y = diopts

    # Filter the list to keep only values above -5 and below 5
    y = list(filter(lambda x: -5 < x < 5, original_y))

    # Create a new list with 1 where the number is preserved and 0 where it was deleted
    preserved_img = [1 if x in y else 0 for x in original_y]

    # Klein Rechner
    folder_path = r"C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\masks"
    x_masks = []
    img_num = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".jpg") and preserved_img[img_num]:
            img_path = os.path.join(folder_path, filename)
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_array is not None:
                x_masks.append(img_array)
        img_num += 1
    x_train_masks = x_masks[:51]
    x_val_masks   = x_masks[51:58]
    x_test_masks  = x_masks[58:]

    y_normalized = [(val - min(y)) / (max(y) - min(y)) for val in y] # Normalize Drioptrieäanderung from -5 to 5 into 0 to 1

    y_train       = y[:51]
    y_val         = y[51:58]
    y_test        = y[58:]

    x_abs_size = []
    for mask in x_masks:
        abs_size = 0
        for row in mask:
            # Sum only the values equal to 255
            row_sum = sum(1 for value in row if value == 255)
            abs_size += row_sum
        x_abs_size.append(abs_size)
    x_rel_size = [i / (x_masks[0].shape[0] * x_masks[0].shape[1]) for i in x_abs_size] # Relative size of each bubble, normalized between 0 and 1

    # Was fehlt: further divide x_rel_size into training, validation, and test set

    plt.scatter(x_rel_size, y_normalized, color='b', label='Data points')
    plt.xlabel('Relative Size')
    plt.ylabel('Dioptrie Change')  # Replace with your actual label
    plt.title('Scatter Plot of Relative Size vs. Normalized Dioptrie Change')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 1)  # Set x-axis limits
    plt.ylim(0, 1)  # Set y-axis limits
    plt.show()

    plt.scatter(x_abs_size, y, color='b', label='Data points')
    plt.xlabel('Relative Size')
    plt.ylabel('Dioptrie Change')  # Replace with your actual label
    plt.title('Scatter Plot of Absolute Size vs. Real Dioptrie Change')
    plt.legend()
    plt.grid(True)
    #plt.xlim(0, 1)  # Set x-axis limits
    #plt.ylim(0, 1)  # Set y-axis limits
    plt.show()

    x_not_normalized, x_normalized, y_not_normalized, y_normalized = x_abs_size, x_rel_size, y, y_normalized

    statistics = Stats()
    normal_distribution_x_not_normalized = statistics.normal_distribution(x_not_normalized)
    normal_distribution_y_not_normalized = statistics.normal_distribution(y_not_normalized)
    normal_distribution_x_normalized     = statistics.normal_distribution(x_normalized)
    normal_distribution_y_normalized     = statistics.normal_distribution(y_normalized)

    variance_x_not_normalized = statistics.variance(x_not_normalized)
    variance_y_not_normalized = statistics.variance(y_not_normalized)
    variance_x_normalized     = statistics.variance(x_normalized)
    variance_y_normalized     = statistics.variance(y_normalized)

    p_value_not_normalized = statistics.t_test(x_not_normalized, y_not_normalized, variance_x_not_normalized == variance_y_not_normalized)
    p_value_normalized     = statistics.t_test(x_normalized, y_normalized, variance_x_normalized == variance_y_normalized)

    print('Hello world')