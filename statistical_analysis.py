# statistical analysis
import cv2
import os
import matplotlib.pyplot as plt
import pickle
from scipy.stats import shapiro
import scipy.stats as stats
import numpy as np

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

# Define the paths to the original images folder and the masks folder
original_images_folder = r"C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\original"
bubble_masks_folder    = r"C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\new_masks"
volume_masks_folder    = r"C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\volumen"

# Get the list of image files in both folders
original_images_files = os.listdir(original_images_folder)
bubble_masks_files    = os.listdir(bubble_masks_folder)
volume_masks_files    = os.listdir(volume_masks_folder)

factor = 10 # 10, 12, 15, 18, 20
new_height = 4500 // factor
new_width = 4500 // factor
original_height = new_height * factor
original_width = new_width * factor

with open("test", "rb") as fp:   # Unpickling
    diopts = pickle.load(fp)

original_y = diopts

y = list(filter(lambda x: -5 < x < 5, original_y))

preserved_img = [1 if x in y else 0 for x in original_y]

y_normalized = [(val - min(y)) / (max(y) - min(y)) for val in y] # Normalize Drioptrieäanderung from -5 to 5 into 0 to 1

# Iterate over the original images
img_num = 0
original_images = []
for image_file in original_images_files:
    if preserved_img[img_num]:
        original_image = cv2.imread(os.path.join(original_images_folder, image_file), cv2.IMREAD_GRAYSCALE)
        original_image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        original_images.append(original_image)
    img_num += 1

img_num = 0
bubble_mask_images = []
for mask_file in bubble_masks_files:
    if preserved_img[img_num]:
        bubble_mask_image = cv2.imread(os.path.join(bubble_masks_folder, mask_file), cv2.IMREAD_GRAYSCALE)
        bubble_mask_image = cv2.resize(bubble_mask_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        bubble_mask_images.append(bubble_mask_image)
    img_num += 1

img_num = 0
volume_mask_images = []
for mask_file in volume_masks_files:
    if preserved_img[img_num]:
        volume_mask_image = cv2.imread(os.path.join(volume_masks_folder, mask_file), cv2.IMREAD_GRAYSCALE)
        volume_mask_image = cv2.resize(volume_mask_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        volume_mask_images.append(volume_mask_image)
    img_num += 1

x_abs_size_bubbles = []
for mask in bubble_mask_images: 
    abs_size = 0
    for row in mask:
        row_sum = sum(1 for value in row if value == 255)
        abs_size += row_sum
    x_abs_size_bubbles.append(abs_size)
x_rel_size_bubbles = [i / (bubble_mask_images[0].shape[0] * bubble_mask_images[0].shape[1]) for i in x_abs_size_bubbles] 

# Was fehlt: further divide x_rel_size into training, validation, and test set

plt.scatter(x_rel_size_bubbles, y_normalized, color='b', label='Data points')
plt.xlabel('Relative Size')
plt.ylabel('Dioptrie Change')  # Replace with your actual label
plt.title('Scatter Plot of Relative Size (Bubbles) vs. Normalized Dioptrie Change')
plt.legend()
plt.grid(True)
plt.xlim(0, 1)  # Set x-axis limits
plt.ylim(0, 1)  # Set y-axis limits
plt.show()

plt.scatter(x_abs_size_bubbles, y, color='b', label='Data points')
plt.xlabel('Pixel Size')
plt.ylabel('Dioptrie Change')  # Replace with your actual label
plt.title('Scatter Plot of Absolute Size (Bubbles) vs. Real Dioptrie Change')
plt.legend()
plt.grid(True)
    #plt.xlim(0, 1)  # Set x-axis limits
    #plt.ylim(0, 1)  # Set y-axis limits
plt.show()

x_not_normalized, x_normalized, y_not_normalized, y_normalized = x_abs_size_bubbles, x_rel_size_bubbles, y, y_normalized

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

x_abs_size_volume = []
for mask in volume_mask_images: 
    abs_size = 0
    for row in mask:
        row_sum = sum(1 for value in row if value == 255)
        abs_size += row_sum
    x_abs_size_volume.append(abs_size)
x_rel_size_volume = [i / (volume_mask_images[0].shape[0] * volume_mask_images[0].shape[1]) for i in x_abs_size_volume] 

    # Was fehlt: further divide x_rel_size into training, validation, and test set

plt.scatter(x_rel_size_volume, y_normalized, color='b', label='Data points')
plt.xlabel('Relative Size')
plt.ylabel('Dioptrie Change')  # Replace with your actual label
plt.title('Scatter Plot of Relative Size (Volumen) vs. Normalized Dioptrie Change')
plt.legend()
plt.grid(True)
plt.xlim(0, 1)  # Set x-axis limits
plt.ylim(0, 1)  # Set y-axis limits
plt.show()

plt.scatter(x_abs_size_volume, y, color='b', label='Data points')
plt.xlabel('Relative Size')
plt.ylabel('Dioptrie Change')  # Replace with your actual label
plt.title('Scatter Plot of Absolute Size (Volumen) vs. Real Dioptrie Change')
plt.legend()
plt.grid(True)
    #plt.xlim(0, 1)  # Set x-axis limits
    #plt.ylim(0, 1)  # Set y-axis limits
plt.show()

x_not_normalized, x_normalized, y_not_normalized, y_normalized = x_abs_size_volume, x_rel_size_volume, y, y_normalized

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

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(x_abs_size_bubbles, x_abs_size_volume, y, marker='o')

ax.set_xlabel('bubbles')
ax.set_ylabel('volume')
ax.set_zlabel('dioptre')

plt.show()

print('Hello world')