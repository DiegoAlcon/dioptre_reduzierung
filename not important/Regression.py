# This programs aims to compute Multiple Linear Regression among a number of features

import numpy as np
import cv2
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import re

# Define the paths to the original images folder and the masks folder
bubble_masks_folder    = r"C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\bubbles"
volume_masks_folder    = r"C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\volumen"
segmen_masks_folder    = r"C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung\segm"

# Get the list of image files in both folders
bubble_masks_files    = os.listdir(bubble_masks_folder)
volume_masks_files    = os.listdir(volume_masks_folder)
segmen_masks_files    = os.listdir(segmen_masks_folder)

r = re.compile(r'\d+')
bubble_masks_files.sort(key=lambda x: int(r.search(x).group()))
volume_masks_files.sort(key=lambda x: int(r.search(x).group()))
segmen_masks_files.sort(key=lambda x: int(r.search(x).group()))

factor = int(input('Enter factor for downsamplig (possibe options: 10, 12, 15, 18, 20): '))
new_height = 4500 // factor
new_width = 4500 // factor
original_height = new_height * factor
original_width = new_width * factor

with open("test", "rb") as fp:   
    diopts = pickle.load(fp)

original_y = diopts

#y = list(filter(lambda x: -5 < x < 5, original_y)) # to filter dioptre to conserve values just from -5 to 5
y = list(filter(lambda x: -10 < x < 0, original_y)) # to filter dioptre to conserve values just negative
#y = original_y # to not filter dioptre

preserved_img = [1 if x in y else 0 for x in original_y]

img_num = 0
bubble_mask_images = []
for mask_file in bubble_masks_files:
    if preserved_img[img_num]:
        bubble_mask_image = cv2.imread(os.path.join(bubble_masks_folder, mask_file), cv2.IMREAD_GRAYSCALE)
        bubble_mask_image = cv2.resize(bubble_mask_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        bubble_mask_images.append(bubble_mask_image)
    img_num += 1

#img_num = 0
#volume_mask_images = []
#for mask_file in volume_masks_files:
#    if preserved_img[img_num]:
#        volume_mask_image = cv2.imread(os.path.join(volume_masks_folder, mask_file), cv2.IMREAD_GRAYSCALE)
#        volume_mask_image = cv2.resize(volume_mask_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
#        volume_mask_images.append(volume_mask_image)
#    img_num += 1

img_num = 0
segmen_mask_images = []
for mask_file in segmen_masks_files:
    if preserved_img[img_num]:
        segmen_mask_image = cv2.imread(os.path.join(segmen_masks_folder, mask_file), cv2.IMREAD_GRAYSCALE)
        segmen_mask_image = cv2.resize(segmen_mask_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        segmen_mask_images.append(segmen_mask_image)
    img_num += 1

x_abs_size_bubbles = []
for mask in bubble_mask_images: 
    abs_size = 0
    for row in mask:
        row_sum = sum(1 for value in row if value == 255)
        abs_size += row_sum
    x_abs_size_bubbles.append(abs_size)

#x_abs_size_volume = []
#for mask in volume_mask_images: 
#    abs_size = 0
#    for row in mask:
#        row_sum = sum(1 for value in row if value == 255) 
#        abs_size += row_sum
#    x_abs_size_volume.append(abs_size)

x_abs_size_segmen = []
for mask in segmen_mask_images: 
    abs_size = 0
    for row in mask:
        row_sum = sum(1 for value in row if value == 255)
        abs_size += row_sum
    x_abs_size_segmen.append(abs_size)

# Create a DataFrame with absolute values
n = 0.8
#df_train = pd.DataFrame({
#    "bubbles": x_abs_size_bubbles[:int(np.round(n*len(y)))],
#    "volume": x_abs_size_volume[:int(np.round(n*len(y)))],
#    "segmen": x_abs_size_segmen[:int(np.round(n*len(y)))],
#    "dioptre": y[:int(np.round(n*len(y)))]
#})
#df_test = pd.DataFrame({
#    "bubbles": x_abs_size_bubbles[int(np.round(n*len(y))):],
#    "volume": x_abs_size_volume[int(np.round(n*len(y))):],
#    "segmen": x_abs_size_segmen[int(np.round(n*len(y))):],
#    "dioptre": y[int(np.round(n*len(y))):]
#})

df_train = pd.DataFrame({
    "segmen": x_abs_size_segmen[:int(np.round(n*len(y)))],
    "bubbles": x_abs_size_bubbles[:int(np.round(n*len(y)))],
    "dioptre": y[:int(np.round(n*len(y)))]
})
df_test = pd.DataFrame({
    "segmen": x_abs_size_segmen[int(np.round(n*len(y))):],
    "bubbles": x_abs_size_bubbles[int(np.round(n*len(y))):],
    "dioptre": y[int(np.round(n*len(y))):]
})

degree = int(input('Enter degree (integer) of polynomial to adjust the data (must be >= 1): '))

poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(df_train[["segmen", "bubbles"]])

# Fit a multiple regression model
model = LinearRegression()
model.fit(X_poly, df_train["dioptre"])

# Get coefficients
coefficients = model.coef_
intercept = model.intercept_

# Predictions
X_poly_test = poly.fit_transform(df_test[["segmen" ,"bubbles"]])
y_pred = model.predict(X_poly_test)

# Compute metrics
mse = mean_squared_error(df_test["dioptre"], y_pred)
r2 = r2_score(df_test["dioptre"], y_pred)

print(f"Coefficients: {coefficients}")
print(f"Intercept: {intercept}")
print(f"MSE: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

# Scatter plot of original vs predicted data
plt.figure(figsize=(10, 6))
plt.scatter(df_test["dioptre"], y_pred, label="Original Data", color="blue")
plt.plot([min(df_test["dioptre"]), max(df_test["dioptre"])], [min(df_test["dioptre"]), max(df_test["dioptre"])], color='r', linestyle='--', label='Ideal Line')
plt.xlabel("Oiriginal value")
plt.ylabel("Predicted value")
plt.title("Original vs. Predicted")
plt.legend()
plt.show()

# Create meshgrid for surface plot
x_vals = np.linspace(df_test["segmen"].min(), df_test["segmen"].max(), 100)
y_vals = np.linspace(df_test["bubbles"].min(), df_test["bubbles"].max(), 100)
X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)
Z_mesh = model.predict(poly.transform(pd.DataFrame({"segmen": X_mesh.ravel(), "bubbles": Y_mesh.ravel()})))
    
# Surface plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(df_test["segmen"], df_test["bubbles"], df_test["dioptre"], label="Original Data", color="blue")
ax.scatter(df_test["segmen"], df_test["bubbles"], y_pred, label="Predicted Data", color="red")
ax.plot_surface(X_mesh, Y_mesh, Z_mesh.reshape(X_mesh.shape), alpha=0.5, cmap="viridis")
ax.set_xlabel("segmen")
ax.set_ylabel("bubbles")
ax.set_zlabel("Dioptre")
ax.set_title("Surface Plot with Original Data (Dioptre)")
plt.legend()
plt.show()

print('Hello world')