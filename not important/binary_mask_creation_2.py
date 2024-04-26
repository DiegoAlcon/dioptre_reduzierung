# Create a binary mask of each image
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
#from PIL import Image
import cv2
#from sklearn.model_selection import train_test_split

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
            #self.binary_mask[px, py] = 1
            print(f"Clicked at (x, y): ({px}, {py})")

    def expand_coords(self):
        expanded_x = []
        expanded_y = []

        # Iterate through each pair of adjacent coordinates
        for i in range(len(self.px)):
            # Add the current coordinate to the expanded lists
            expanded_x.append(self.px[i])
            expanded_y.append(self.py[i])

            # Interpolate between adjacent coordinates
            dx = self.px[(i + 1) % len(self.px)] - self.px[i]
            dy = self.py[(i + 1) % len(self.py)] - self.py[i]

            # Calculate the number of steps needed for interpolation
            steps = max(abs(dx), abs(dy))

            # Interpolate and add intermediate coordinates
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

if __name__ == "__main__":
    # C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\masks
    image_directory = [r"C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\Labeled1", 
                       r"C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\Labeled2", 
                       ]  
    excel_directory = "example.xlsx"
    image_processor = Bildvorverarbeitung(image_directory, excel_directory, target_height=900, target_width=900, x_offset=-225, y_offset=1250) # 850 850 -225 1250

    images = image_processor.images[0]
    diopts = image_processor.images[1]

    images = image_processor.crop_images(images)

    #image_plotter = BildPlotter(images) 
    #image_plotter.plot_image(2) # 1 soll images index werden, 2 darf es nicht

    # Here Canny and binarization

    del images[55]
    del diopts[55]

    factor = 5
    new_height = images[0].shape[0] // factor
    new_width = images[0].shape[1] // factor
    original_height = new_height * factor
    original_width = new_width * factor
    #images = [cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA) for img in images]

    x = images
    y = diopts

    train_size = int(0.8 * len(x))
    x_train, x_temp = x[:train_size], x[train_size:]
    test_size = int(0.5 * len(x_temp))
    x_val, x_test = x_temp[:test_size], x_temp[test_size:]

    x_train_masks = []
    file_number = 0
    for image in x_train:
        original_image = image
        image = binary_mask = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        file_number += 1
        by_instance = Masking()

        print(x_train_masks)

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

        x_train_masks.append(binary_mask)     

        binary_mask = cv2.resize(binary_mask, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

        #image_plotter = BildPlotter(binary_mask) 
        #image_plotter.plot_image(1) 

        result = cv2.bitwise_and(original_image, original_image, mask=binary_mask)

        image_plotter = BildPlotter(result) 
        image_plotter.plot_image(1)         

        output_path = os.path.join(r'C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\masks', f"mask_{file_number}.jpg")
        plt.imsave(output_path, binary_mask, cmap='gray', pil_kwargs={'compress_level': 0})

    image_plotter = BildPlotter(x_train_masks) 
    image_plotter.plot_image(2) 
    
    print('Hello world')