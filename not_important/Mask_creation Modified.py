# This programs aims to be a GUI for mask creation

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import cv2

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
            fig, axs = plt.subplots(8, 12, figsize=(15, 15))

            for i in range(8):
                for j in range(12):
                    index = i * 12 + j
                    if index < len(self.images):
                        axs[i, j].imshow(self.images[index], cmap='gray')
                        axs[i, j].axis("off") 

            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            plt.show()

class Masking():
    
    def __init__(self):
        self.px = []
        self.py = []

    def handle_click(self, event):
        if event.xdata is not None and event.ydata is not None:
            px, py = int(event.xdata), int(event.ydata)
            self.px.append(px)
            self.py.append(py)
            print(f"Clicked at (x, y): ({px}, {py})")

    def expand_coords(self):
        expanded_x = []
        expanded_y = []

        for i in range(len(self.px)):
            expanded_x.append(self.px[i])
            expanded_y.append(self.py[i])

            dx = self.px[(i + 1) % len(self.px)] - self.px[i]
            dy = self.py[(i + 1) % len(self.py)] - self.py[i]

            steps = max(abs(dx), abs(dy))

            for j in range(1, steps):
                interp_x = self.px[i] + (dx * j) // steps
                interp_y = self.py[i] + (dy * j) // steps
                expanded_x.append(interp_x)
                expanded_y.append(interp_y)

        return expanded_x, expanded_y
    
    def cauchy_argument_principle(self, expanded_x, expanded_y, px, py):
        n = len(expanded_x)
        winding_number = 0

        for i in range(n):
            xi, yi = expanded_x[i], expanded_y[i]
            xj, yj = expanded_x[(i + 1) % n], expanded_y[(i + 1) % n]

            if (yi <= py < yj or yj <= py < yi) and px <= max(xi, xj):
                intersection_x = (py - yi) * (xj - xi) / (yj - yi) + xi

                if px < intersection_x:
                    winding_number += 1

        return winding_number % 2 == 1

if __name__ == "__main__":
    # Kleiner Rechner
    images_folder = r'C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\original' 
    # Mittlere Rechner
    # Grösste Rechner
    
    image_files  = os.listdir(images_folder)

    new_height = 256
    new_width = 256
    original_height = 4504
    original_width = 4504

    images = []
    for image_file in image_files:
        image = cv2.imread(os.path.join(images_folder, image_file), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        images.append(image)

    #########################################################################
    # From this position forward, the images are masked and stored

    bubbles_to_store = []
    segmens_to_store = []

    features = 1 # Change the features to capture, 1 for bubbles, 2 for segments
    image2mask = 1

    ab = int(input('From which image to, should the masking be performed?: ')) # Enter 96 for new masking
    images = images[ab:]

    for image in images:
        x_masks = []
        while image2mask != 0:
            image = binary_mask = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            by_instance = Masking()
            fig, ax = plt.subplots()
            ax.imshow(image, cmap='gray')
            cid = fig.canvas.mpl_connect('button_press_event', lambda event: by_instance.handle_click(event))
            plt.show()
            expanded_x, expanded_y = by_instance.expand_coords()

            binary_mask = np.zeros_like(image, dtype=np.uint8)

            
            for cols in range(image.shape[0]):
                for rows in range(image.shape[1]):
                    is_inside = by_instance.cauchy_argument_principle(expanded_x, expanded_y, rows, cols)
                    print(is_inside)
                    if is_inside:
                        binary_mask[cols, rows] = 1

            x_masks.append(binary_mask)  

            binary_mask = cv2.resize(binary_mask, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
            if features == 1:
                bubbles_to_store.append(binary_mask)
            elif features == 2:   
                segmens_to_store.append(binary_mask)

            result = cv2.bitwise_and(image, image, mask=binary_mask)
            image_plotter = BildPlotter(result) 
            image_plotter.plot_image(1)         

            if features == 1:
                output_path = os.path.join(r'C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\bubbles', f"mask_{image2mask + 1}.jpg")
                plt.imsave(output_path, binary_mask, cmap='gray', pil_kwargs={'compress_level': 0})
            elif features == 2:
                output_path = os.path.join(r'C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\segm', f"segm_{image2mask + 1}.jpg")
                plt.imsave(output_path, binary_mask, cmap='gray', pil_kwargs={'compress_level': 0})
            

print('Hello world')