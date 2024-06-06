# Heat map & correlation

# Multiple Non Linear Regression
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

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

x_abs_size_volume = []
for mask in volume_mask_images: 
    abs_size = 0
    for row in mask:
        row_sum = sum(1 for value in row if value == 255)
        abs_size += row_sum
    x_abs_size_volume.append(abs_size)


# Create a DataFrame
df = pd.DataFrame({
    "bubbles": x_abs_size_bubbles,
    "volume": x_abs_size_volume,
    "dioptre": y
})

model_type = 3

if model_type == 1:

    # Compute the correlation matrix
    correlation_matrix = df.corr()

    # Create a heatmap using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", square=True)
    plt.title("Correlation Matrix Heatmap")
    plt.show()

    # Separate independent variables (features) and dependent variable (target)
    X = df[['bubbles', 'volume']]
    y = df['dioptre']

    # Initialize the linear regression model
    model = LinearRegression()

    # Fit the model to the data
    model.fit(X, y)

    # Get the coefficients (weights) for each feature
    a = model.coef_[0]  # Weight for 'bubbles'
    b = model.coef_[1]  # Weight for 'volume'

    # Print the coefficients
    print(f"Weight for 'bubbles': {a:.2f}")
    print(f"Weight for 'volume': {b:.2f}")

    # Generate predictions using the model
    df['predicted_dioptre'] = model.predict(X)

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data points
    ax.scatter(df['bubbles'], df['volume'], df['dioptre'], c='b', label='Actual Dioptre')
    ax.scatter(df['bubbles'], df['volume'], df['predicted_dioptre'], c='r', marker='x', label='Predicted Dioptre')

    # Plot the regression plane
    bubbles_range = np.linspace(df['bubbles'].min(), df['bubbles'].max(), 100)
    volume_range = np.linspace(df['volume'].min(), df['volume'].max(), 100)
    bubbles_grid, volume_grid = np.meshgrid(bubbles_range, volume_range)
    dioptre_plane = a * bubbles_grid + b * volume_grid
    #ax.plot_surface(bubbles_grid, volume_grid, dioptre_plane, alpha=0.5, cmap='viridis')

    # Set labels
    ax.set_xlabel('Bubbles')
    ax.set_ylabel('Volume')
    ax.set_zlabel('Dioptre')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.title('Linear Regression Plane for Dioptre Prediction')
    plt.show()

    # Compute MSE
    mse = mean_squared_error(df['dioptre'], df['predicted_dioptre'])

    # Compute R2 score
    r2 = r2_score(df['dioptre'], df['predicted_dioptre'])

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2) Score: {r2:.2f}")

elif model_type == 2:
    # Fit a quadratic model
    X = df[['bubbles', 'volume']]
    X['bubbles_sq'] = X['bubbles'] ** 2
    X['volume_sq'] = X['volume'] ** 2
    y = df['dioptre']

    model = LinearRegression()
    model.fit(X, y)

    # Get coefficients
    coef_b0 = model.intercept_
    coef_b1, coef_b2, coef_b3, coef_b4 = model.coef_

    # Predict the data
    df['predicted_dioptre'] = model.predict(X)

    # Compute metrics
    mse = mean_squared_error(y, df['predicted_dioptre'])
    r2 = r2_score(y, df['predicted_dioptre'])

    # Create a meshgrid for the surface plot
    bubbles_range = np.linspace(df['bubbles'].min(), df['bubbles'].max(), 100)
    volume_range = np.linspace(df['volume'].min(), df['volume'].max(), 100)
    bubbles_grid, volume_grid = np.meshgrid(bubbles_range, volume_range)

    # Compute the predicted dioptre values for the meshgrid
    dioptre_surface = (
        coef_b0 +
        coef_b1 * bubbles_grid +
        coef_b2 * volume_grid +
        coef_b3 * bubbles_grid**2 +
        coef_b4 * volume_grid**2
    )

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df['bubbles'], df['volume'], df['dioptre'], label='Original Data', c='b', marker='o')
    ax.scatter(df['bubbles'], df['volume'], df['predicted_dioptre'], label='Predicted Data', c='r', marker='x')

    # Plot the surface
    ax.plot_surface(bubbles_grid, volume_grid, dioptre_surface, alpha=0.5, cmap='viridis')

    ax.set_xlabel('Bubbles')
    ax.set_ylabel('Volume')
    ax.set_zlabel('Dioptre')
    ax.set_title('3D Scatter Plot with Surface - Original vs. Predicted Data')
    ax.legend()

    plt.show()

    # Print coefficients and metrics
    print(f"Coefficients: b0 = {coef_b0:.2f}, b1 = {coef_b1:.2f}, b2 = {coef_b2:.2f}, b3 = {coef_b3:.2f}, b4 = {coef_b4:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")

elif model_type == 3:

    degree = 1  # Change to desired degree
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(df[["bubbles", "volume"]])

    # Fit a multiple regression model
    model = LinearRegression()
    model.fit(X_poly, df["dioptre"])

    # Get coefficients
    coefficients = model.coef_
    intercept = model.intercept_

    # Predictions
    y_pred = model.predict(X_poly)

    # Compute metrics
    mse = mean_squared_error(df["dioptre"], y_pred)
    r2 = r2_score(df["dioptre"], y_pred)

    print(f"Coefficients: {coefficients}")
    print(f"Intercept: {intercept}")
    print(f"MSE: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")

    # Scatter plot of original vs predicted data
    plt.figure(figsize=(10, 6))
    plt.scatter(df["dioptre"], y_pred, label="Original Data", color="blue")
    plt.plot([min(df["dioptre"]), max(df["dioptre"])], [min(df["dioptre"]), max(df["dioptre"])], color='r', linestyle='--', label='Ideal Line')
    plt.xlabel("Oiriginal value")
    plt.ylabel("Predicted value")
    plt.title("Original vs. Predicted")
    plt.legend()
    plt.show()

    # Create meshgrid for surface plot
    x_vals = np.linspace(df["bubbles"].min(), df["bubbles"].max(), 100)
    y_vals = np.linspace(df["volume"].min(), df["volume"].max(), 100)
    X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)
    Z_mesh = model.predict(poly.transform(pd.DataFrame({"bubbles": X_mesh.ravel(), "volume": Y_mesh.ravel()})))
    
    # Surface plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(df["bubbles"], df["volume"], df["dioptre"], label="Original Data", color="blue")
    ax.scatter(df["bubbles"], df["volume"], y_pred, label="Predicted Data", color="red")
    ax.plot_surface(X_mesh, Y_mesh, Z_mesh.reshape(X_mesh.shape), alpha=0.5, cmap="viridis")
    ax.set_xlabel("Bubbles")
    ax.set_ylabel("Volume")
    ax.set_zlabel("Dioptre")
    ax.set_title("Surface Plot with Original Data (Dioptre)")
    plt.legend()
    plt.show()

print('Hello world')