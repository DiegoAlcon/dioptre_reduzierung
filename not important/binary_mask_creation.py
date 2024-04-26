# Create a binary mask of each image
import matplotlib.pyplot as plt
import numpy as np

# Load your image (replace 'your_image.jpg' with the actual image file)


image = plt.imread('sample_image.jpg')

# Initialize an empty binary mask (same size as the image)
binary_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)

def onclick(event):
    # Get the pixel coordinates where the user clicked
    x, y = int(event.xdata), int(event.ydata)
    
    # Set the corresponding pixel in the mask to 1 (object)
    binary_mask[y, x] = 1
    
    # Update the display (optional)
    plt.imshow(image)
    plt.imshow(binary_mask, alpha=0.3, cmap='Greens')
    plt.draw()

# Display the image
fig, ax = plt.subplots()
ax.imshow(image)
cid = fig.canvas.mpl_connect('button_press_event', onclick)
#cid = fig.canvas.mpl_connect('motion_notify_event', onclick)

plt.show()

print('Hello World')

# Now 'binary_mask' contains the annotated mask
# Save it or use it for further processing
