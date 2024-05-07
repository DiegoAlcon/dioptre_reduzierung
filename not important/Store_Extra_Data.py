# This program is not officially part of the 7 important, is just to create a folder 
# with those 32 images with ordered diopters and save them all there

# This program aims to perform the datapreparation stage and store them all in defined folders

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import cv2
import pickle
import glob

class Bildvorverarbeitung:
    def __init__(self, image_directory, excel_directory, ordered_filenames, target_height, target_width, x_offset, y_offset): 
        self.image_directory   = image_directory
        self.excel_directory   = excel_directory 
        self.ordered_filenames = ordered_filenames                                                            
        self.target_height     = target_height                                    
        self.target_width      = target_width
        self.x_offset          = x_offset
        self.y_offset          = y_offset
        self.images            = self.load_images_from_directory_call()

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
        diopt_abs = df['OPTIC OF - Diopter'].tolist()

        for filename in self.ordered_filenames:
            base_name, ext = os.path.splitext(filename)
            parts = base_name.split("_")
            title_img = parts[2][:6]
            sample_img = parts[2][7:]
            idx_title = [i for i, a in enumerate(title_exc) if a == title_img]
            idx_sample   = [i for i, a in enumerate(sample_exc) if a == int(sample_img)]

            idx = np.intersect1d(idx_title, idx_sample)[0] 

            diopt_delta = diopt_abs[idx]

            if os.path.isfile(os.path.join(image_directory[0],filename)):
                images.append(cv2.imread(os.path.join(image_directory[0], filename), cv2.IMREAD_GRAYSCALE))
            elif os.path.isfile(os.path.join(image_directory[1],filename)):
                images.append(cv2.imread(os.path.join(image_directory[1], filename), cv2.IMREAD_GRAYSCALE))
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
    
if __name__ == "__main__":
    # Kleiner Rechner
    image_directory = [r"C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\Labeled1_18_4", 
                       r"C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\Labeled2_18_4", 
                       ] 
    # Mittlerer Rechner
    #image_directory = [r"C:\Users\SANCHDI2\dioptre_reduzierung\Labeled1_18_4",
    #                   r"C:\Users\SANCHDI2\dioptre_reduzierung\Labeled2_18_4"
    #                    ]
    with open("common_filenames", "rb") as fp:   
        common_filenames = pickle.load(fp)
    with open("unique_filenames", "rb") as fp:
        unique_filenames = pickle.load(fp)

    common_filenames = np.array(common_filenames)
    unique_filenames = np.array(unique_filenames)
    ordered_filenames = np.concatenate((common_filenames, unique_filenames), axis=0)

    excel_directory = "example.xlsx"
    image_processor = Bildvorverarbeitung(image_directory, excel_directory, ordered_filenames, target_height=4500, target_width=4500, x_offset=0, y_offset=0) 

    images = image_processor.images[0]
    diopts = image_processor.images[1]

    images = image_processor.crop_images(images)

    # Klein Rechner
    directory = r'C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\original\*'
    # Mittlere Rechner
    #directory = r"C:\Users\SANCHDI2\dioptre_reduzierung\original\*"
    files = glob.glob(directory)

    for file in files:
        if os.path.exists(file):
            os.remove(file)

    file_number = 0
    for img in images:
        file_number += 1
        # Klein Rechner
        output_path = os.path.join(r'C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\original', f"original_{file_number}.jpg")
        # Mittlere Rechner
        #output_path = os.path.join(r"C:\Users\SANCHDI2\dioptre_reduzierung\original", f"original_{file_number}.jpg")
        plt.imsave(output_path, img, cmap='gray', pil_kwargs={'compress_level': 0})

    with open("test", "wb") as fp:   
        pickle.dump(diopts, fp)