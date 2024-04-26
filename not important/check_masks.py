# Check if volum mask was good

import cv2
import os
import matplotlib.pyplot as plt

# Define the paths to the original images folder and the masks folder
original_images_folder = r"C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\original"
masks_folder = r"C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\volumen"

# Get the list of image files in both folders
original_images_files = os.listdir(original_images_folder)
masks_files = os.listdir(masks_folder)

# Iterate over the original images
original_images_ = []
for image_file in original_images_files:
    original_image = cv2.imread(os.path.join(original_images_folder, image_file), cv2.IMREAD_GRAYSCALE)
    original_images_.append(original_image)

mask_images_ = []
for mask_file in masks_files:
    mask_image = cv2.imread(os.path.join(masks_folder, mask_file), cv2.IMREAD_GRAYSCALE)
    mask_images_.append(mask_image)


# Iterate over each pair of original image and mask
#for i, (original_image, mask_image) in enumerate(zip(original_images_, mask_images_), 1):
#    # Apply the mask to the original image
#    masked_image = cv2.bitwise_and(original_image, original_image, mask=mask_image)

#    # Display the masked image
#    #plt.subplot(2, len(original_images_), i)
#    plt.imshow(masked_image, cmap="gray")
#    plt.title(f"Masked Image {i-1}")
#    plt.axis("off")
#    plt.show()

#    print('Hello World')

fig, axs = plt.subplots(8, 12, figsize=(15, 15))

for i in range(8):
    for j in range(12):
        index = i * 12 + j
        if index < len(mask_images_):
            axs[i, j].imshow(mask_images_[index], cmap='gray')
            axs[i, j].axis("off") 

plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.show()
