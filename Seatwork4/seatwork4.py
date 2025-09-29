import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('image.jpg')
if image is None:
    raise ValueError("Image not found. Please provide a valid image path.")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Harris corner detection
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# Dilate the corner response to make corners more visible
dst = cv2.dilate(dst, None)

# Threshold to select strong corners
threshold = 0.01 * dst.max()
# Create a copy of the original image to mark corners
output_image = image.copy()

# Mark corners in red (BGR: [0, 0, 255]) where dst > threshold
output_image[dst > threshold] = [0, 0, 255]

# Convert BGR to RGB for correct color display in matplotlib
output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

# Display the result using matplotlib
plt.figure(figsize=(8, 6))
plt.imshow(output_image_rgb)
plt.title('Harris Corners')
plt.axis('off')  # Hide axes
plt.show()

# Save the result
cv2.imwrite('harris_corners_output.jpg', output_image)
print("Output saved as 'harris_corners_output.jpg'.")