import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_circles_and_ellipses(image_path):
    # Read the image from the given file path
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Hough Transform to detect circles
    circles = cv2.HoughCircles(
        gray_image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=50
    )

    # If circles are detected, draw them on the image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            cv2.circle(image, center, radius, (0, 0, 255), 2)

    # Detect ellipses using the Hough Ellipse Transform
    ellipses = cv2.HoughCircles(
        gray_image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=50
    )

    # If ellipses are detected, draw them on the image
    if ellipses is not None:
        ellipses = np.uint16(np.around(ellipses))
        for ellipse in ellipses[0, :]:
            center = (ellipse[0], ellipse[1])
            major_axis = ellipse[2]
            minor_axis = ellipse[3]
            angle = ellipse[4]
            cv2.ellipse(image, center, (major_axis, minor_axis), angle, 0, 360, (0, 255, 0), 2)

    # Display the image with detected circles and ellipses
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Detected Circles and Ellipses")
    plt.axis("off")
    plt.show()

# Example usage: replace "path/to/your/image.jpg" with the actual image path
image_path = r"C:\Users\SANCHDI2\OneDrive - Alcon\Desktop\Circle Images\circles1.jpg"
detect_circles_and_ellipses(image_path)
