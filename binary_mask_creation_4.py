# Create a binary mask of each image
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import cv2
from scipy.stats import shapiro
import scipy.stats as stats

# This program is altered such that only a couple of images are re-masked

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
            fig, axs = plt.subplots(8, 12, figsize=(15, 15))

            for i in range(8):
                for j in range(12):
                    index = i * 12 + j
                    if index < len(self.images):
                        axs[i, j].imshow(self.images[index], cmap='gray')
                        axs[i, j].axis("off") 

            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            plt.show()

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

        number_of_images_in_folders = 0
        number_of_images_in_folders_and_excel = 0
        number_of_images_in_folders_and_excel_with_data = 0

        for directory in self.image_directory:
            directory_filenames = os.listdir(directory)
            for filename in directory_filenames:
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    number_of_images_in_folders += 1
                    base_name, ext = os.path.splitext(filename)
                    parts = base_name.split("_")

                    title_img = parts[2][:6]
                    sample_img = parts[2][7:]

                    idx_title = [i for i, a in enumerate(title_exc) if a == title_img]
                    idx_sample   = [i for i, a in enumerate(sample_exc) if a == int(sample_img)]

                    if len(idx_title) != 0 and len(idx_sample) != 0:
                        if len(np.intersect1d(idx_title, idx_sample)) != 0:
                            number_of_images_in_folders_and_excel += 1
                            idx = np.intersect1d(idx_title, idx_sample)[0] 
                        else:
                            continue
                        if not(np.isnan(diopt_post[idx])) and not(np.isnan(diopt_pre[idx])):
                            number_of_images_in_folders_and_excel_with_data += 1
                            diopt_delta = diopt_post[idx] - diopt_pre[idx]
                        else:
                            continue
                    else:
                        continue

                    images.append(cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE))
                    diopt.append(diopt_delta)

        print('Number of images in folders: ', number_of_images_in_folders)
        print('Number of images in folders and excel: ', number_of_images_in_folders_and_excel)
        print('Number of images in folders and excel with data: ', number_of_images_in_folders_and_excel_with_data)
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
        
        # Check if lists were good expanded:
        #plt.scatter(expanded_x, expanded_y, color='b', label='Data points')
        #plt.show()

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
    # Kleiner Rechner
    image_directory = [r"C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\Labeled1_18_4", 
                       r"C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\Labeled2_18_4", 
                       ] 
    excel_directory = "example.xlsx"
    image_processor = Bildvorverarbeitung(image_directory, excel_directory, target_height=4500, target_width=4500, x_offset=0, y_offset=0) # 850 850 -225 1250

    images = image_processor.images[0]
    diopts = image_processor.images[1]

    images = image_processor.crop_images(images)

    #image_plotter = BildPlotter(images) 
    #image_plotter.plot_image(2) # 1 soll images index werden, 2 darf es nicht

    #del images[55]
    #del diopts[55]

    factor = 10 # 10, 12, 15, 18, 20
    new_height = images[0].shape[0] // factor
    new_width = images[0].shape[1] // factor
    original_height = new_height * factor
    original_width = new_width * factor

    x = images
    y = diopts

    train_size = int(0.8 * len(x))
    x_train, x_temp = x[:train_size], x[train_size:]
    test_size = int(0.5 * len(x_temp))
    x_val, x_test = x_temp[:test_size], x_temp[test_size:]

    bubbles_to_store = []
    volumen_to_store = []

    #x = x[84:] # added because an stop

    image2mask = 1

    for features in [2]: # change for needed type of features, 1: bubbles, 2: volume, [1, 2]: bubbles and volume
        x_masks = []
        #file_number = 83 # added because an stop
        while image2mask != -1:
            image2mask = int(input('Enter the image to maks: '))
            original_image = x[image2mask]
            image = binary_mask = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            #file_number += 1
            by_instance = Masking()

            #print(x_masks)

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
                volumen_to_store.append(binary_mask)

            #image_plotter = BildPlotter(binary_mask) 
            #image_plotter.plot_image(1) 

            result = cv2.bitwise_and(original_image, original_image, mask=binary_mask)

            image_plotter = BildPlotter(result) 
            image_plotter.plot_image(1)         

            if features == 1:
                output_path = os.path.join(r'C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\new_masks', f"mask_{image2mask + 1}.jpg")
                plt.imsave(output_path, binary_mask, cmap='gray', pil_kwargs={'compress_level': 0})
            elif features == 2:
                output_path = os.path.join(r'C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\volumen', f"volum_{image2mask + 1}.jpg")
                plt.imsave(output_path, binary_mask, cmap='gray', pil_kwargs={'compress_level': 0})

        image_plotter = BildPlotter(x_masks) 
        image_plotter.plot_image(2) 
    
    original_y = diopts

    y = list(filter(lambda x: -5 < x < 5, original_y))

    preserved_img = [1 if x in y else 0 for x in original_y]

    # Klein Rechner
    #folder_path = r"C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\masks"
    #x_masks = []
    #img_num = 0
    #for filename in os.listdir(folder_path):
    #    if filename.lower().endswith(".jpg") and preserved_img[img_num]:
    #        img_path = os.path.join(folder_path, filename)
    #        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #        if img_array is not None:
    #            x_masks.append(img_array)
    #    img_num += 1
    #x_train_masks = x_masks[:51]
    #x_val_masks   = x_masks[51:58]
    #x_test_masks  = x_masks[58:]

    y_normalized = [(val - min(y)) / (max(y) - min(y)) for val in y] # Normalize Drioptrieäanderung from -5 to 5 into 0 to 1

    y_train       = y[:51]
    y_val         = y[51:58]
    y_test        = y[58:]

    x_abs_size = []
    for mask in bubbles_to_store: 
        abs_size = 0
        for row in mask:
            row_sum = sum(1 for value in row if value == 255)
            abs_size += row_sum
        x_abs_size.append(abs_size)
    x_rel_size = [i / (x_masks[0].shape[0] * x_masks[0].shape[1]) for i in x_abs_size] 

    # Was fehlt: further divide x_rel_size into training, validation, and test set

    plt.scatter(x_rel_size, y_normalized, color='b', label='Data points')
    plt.xlabel('Relative Size')
    plt.ylabel('Dioptrie Change')  # Replace with your actual label
    plt.title('Scatter Plot of Relative Size (Bubbles) vs. Normalized Dioptrie Change')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 1)  # Set x-axis limits
    plt.ylim(0, 1)  # Set y-axis limits
    plt.show()

    plt.scatter(x_abs_size, y, color='b', label='Data points')
    plt.xlabel('Relative Size')
    plt.ylabel('Dioptrie Change')  # Replace with your actual label
    plt.title('Scatter Plot of Absolute Size (Bubbles) vs. Real Dioptrie Change')
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

    print('-------------------------------------------------------------------------------')
    print('Factor: Bubbles') # Change the factor to take into account
    print('The not normalized size of the mask has a normal distribution: ', normal_distribution_x_not_normalized)
    print('The not normalized dioptre change has a normal distribution: ', normal_distribution_y_not_normalized)
    print('The normalized size of the mask has a normal distribution: ', normal_distribution_x_normalized)
    print('The normalized dioptre change has a normal distribution: ', normal_distribution_y_normalized)
    print('The not normalized size of the mask has a variance of: ', variance_x_not_normalized)
    print('The not normalized diopte change has a variance of: ', variance_y_not_normalized)
    print('The normalized size of the mask has a variance of: ', variance_x_normalized)
    print('The normalized dioptre change has a variance of: ', variance_y_normalized)
    print('The P-value of the not normalized events is: ', p_value_not_normalized)
    print('The P_value of the normalized events is: ', p_value_normalized)
    print('-------------------------------------------------------------------------------')

    #######################################################################################################################################

    x_abs_size = []
    for mask in volumen_to_store: 
        abs_size = 0
        for row in mask:
            row_sum = sum(1 for value in row if value == 255)
            abs_size += row_sum
        x_abs_size.append(abs_size)
    x_rel_size = [i / (x_masks[0].shape[0] * x_masks[0].shape[1]) for i in x_abs_size] 

    # Was fehlt: further divide x_rel_size into training, validation, and test set

    plt.scatter(x_rel_size, y_normalized, color='b', label='Data points')
    plt.xlabel('Relative Size')
    plt.ylabel('Dioptrie Change')  # Replace with your actual label
    plt.title('Scatter Plot of Relative Size (Volumen) vs. Normalized Dioptrie Change')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 1)  # Set x-axis limits
    plt.ylim(0, 1)  # Set y-axis limits
    plt.show()

    plt.scatter(x_abs_size, y, color='b', label='Data points')
    plt.xlabel('Relative Size')
    plt.ylabel('Dioptrie Change')  # Replace with your actual label
    plt.title('Scatter Plot of Absolute Size (Volumen) vs. Real Dioptrie Change')
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

    print('-------------------------------------------------------------------------------')
    print('Factor: Volumen') # Change the factor to take into account
    print('The not normalized size of the mask has a normal distribution: ', normal_distribution_x_not_normalized)
    print('The not normalized dioptre change has a normal distribution: ', normal_distribution_y_not_normalized)
    print('The normalized size of the mask has a normal distribution: ', normal_distribution_x_normalized)
    print('The normalized dioptre change has a normal distribution: ', normal_distribution_y_normalized)
    print('The not normalized size of the mask has a variance of: ', variance_x_not_normalized)
    print('The not normalized diopte change has a variance of: ', variance_y_not_normalized)
    print('The normalized size of the mask has a variance of: ', variance_x_normalized)
    print('The normalized dioptre change has a variance of: ', variance_y_normalized)
    print('The P-value of the not normalized events is: ', p_value_not_normalized)
    print('The P_value of the normalized events is: ', p_value_normalized)
    print('-------------------------------------------------------------------------------')

    print('Hello world')