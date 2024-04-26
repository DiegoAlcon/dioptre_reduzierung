import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image from a file path (replace with your own image path)
image_path = r"C:\Users\SANCHDI2\OneDrive - Alcon\Desktop\Circle Images\circles6.png"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Hough Transform to detect circles
circles = cv2.HoughCircles(
    gray_image,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=300,
    param1=50,
    param2=30,
    minRadius=1,
    maxRadius=25
)

# If circles are detected, draw them on the image
if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        center = (circle[0], circle[1])
        radius = circle[2]
        cv2.circle(image, center, radius, (0, 0, 255), 2)

    # Display the image with detected circles
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Detected Circles")
    plt.axis("off")
    plt.show()
else:
    print("No circles detected in the image.")

# Display the original image
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title("Undetected Circles")
# plt.axis("off")
# plt.show()
