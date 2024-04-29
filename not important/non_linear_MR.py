# Multiple Non Linear Regression
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#from sklearn.preprocessing import StandardScaler

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

#y_normalized = [(val - min(y)) / (max(y) - min(y)) for val in y] # Normalize Drioptrieäanderung from -5 to 5 into 0 to 1


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
#x_rel_size_bubbles = [i / (bubble_mask_images[0].shape[0] * bubble_mask_images[0].shape[1]) for i in x_abs_size_bubbles] 

x_abs_size_volume = []
for mask in volume_mask_images: 
    abs_size = 0
    for row in mask:
        row_sum = sum(1 for value in row if value == 255)
        abs_size += row_sum
    x_abs_size_volume.append(abs_size)

train_size_bubbles = int(0.8 * len(x_abs_size_bubbles))
x_train_bubbles, x_test_bubbles = x_abs_size_bubbles[:train_size_bubbles], x_abs_size_bubbles[train_size_bubbles:]
train_size_volume = int(0.8 * len(x_abs_size_volume))
x_train_volume, x_test_volume = x_abs_size_volume[:train_size_volume], x_abs_size_volume[train_size_volume:]
train_size_dioptre = int(0.8 * len(y))
y_train_dioptre, y_test_dioptre = y[:train_size_dioptre], y[train_size_dioptre:]

# Feature scaling ###################################################### SCALE FEATURES
#scaler = StandardScaler()
#x_train_bubbles = scaler.fit_transform(x_train_bubbles)
#x_train_volume  = scaler.fit_transform(x_train_volume)
#x_test_bubbles = scaler.fit_transform(x_test_bubbles)
#x_test_volume = scaler.fit_transform(x_test_volume)
#y_train_dioptre = scaler.fit_transform(y_train_dioptre)
#y_test_dioptre = scaler.fit_transform(y_test_dioptre)

X1 = np.array(x_train_bubbles)
X2 = np.array(x_train_volume)
X_train = np.column_stack((X1, X2))

X1 = np.array(x_test_bubbles)
X2 = np.array(x_test_volume)
X_test = np.column_stack((X1, X2))

Y_train  = np.array(y_train_dioptre)
Y_test   = np.array(y_test_dioptre)

# Create an SVR model with an RBF kernel
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

# Fit the model to the data
svr_rbf.fit(X_train, Y_train)

# Make predictions
Y_pred = svr_rbf.predict(X_test)

# Visualization
#plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap='viridis', label='Actual Data')
#plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_pred, cmap='plasma', marker='x', label='Predictions')
#plt.xlabel('X1')
#plt.ylabel('X2')
#plt.title('SVR with RBF Kernel')
#plt.colorbar(label='y')
#plt.legend()
#plt.show()

plt.scatter(Y_test, Y_pred, alpha=0.5)
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='r', linestyle='--', label='Ideal Line')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Predictions vs. True Values')
plt.show()

# Model evaluation
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

print('Hello world')