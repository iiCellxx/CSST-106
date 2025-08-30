import cv2
import numpy as np

# Load the input image (replace 'input_image.jpg' with your image path)
image = cv2.imread('mugiwara.jpg')

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load the image. Please check the file path.")
    exit()

# Verify image dimensions
if image.shape[:2] != (840, 840):
    print(f"Warning: Image size is {image.shape[:2]}, expected (840, 840).")

# 1. Resize the image to 50% (420x420 pixels)
resized_image = cv2.resize(image, (420, 420), interpolation=cv2.INTER_AREA)

# 2. Crop a centered 400x400 region
center_x, center_y = 840 // 2, 840 // 2  # Image center
crop_size = 400  # Width and height of crop
x_start = center_x - crop_size // 2  # 220
y_start = center_y - crop_size // 2  # 220
cropped_image = image[y_start:y_start + crop_size, x_start:x_start + crop_size]

# 3. Layered outputs: Isolate each color channel
# Create copies of the original image
blue_channel_image = image.copy()
green_channel_image = image.copy()
red_channel_image = image.copy()

# All Blue: Set green and red channels to zero
blue_channel_image[:, :, 1] = 0  # Green channel
blue_channel_image[:, :, 2] = 0  # Red channel

# All Green: Set blue and red channels to zero
green_channel_image[:, :, 0] = 0  # Blue channel
green_channel_image[:, :, 2] = 0  # Red channel

# All Red: Set blue and green channels to zero
red_channel_image[:, :, 0] = 0  # Blue channel
red_channel_image[:, :, 1] = 0  # Green channel

# 4. Edge detection (using Canny edge detector)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_image, 100, 200)  # Optimized thresholds
edge_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel

# Display the results
cv2.imshow('Original Image (840x840)', image)
cv2.imshow('Resized Image (420x420)', resized_image)
cv2.imshow('Cropped Image (400x400)', cropped_image)
cv2.imshow('Blue Channel Image', blue_channel_image)
cv2.imshow('Green Channel Image', green_channel_image)
cv2.imshow('Red Channel Image', red_channel_image)
cv2.imshow('Edge Image', edge_image)

# Save the output images
cv2.imwrite('resized_image.jpg', resized_image)
cv2.imwrite('cropped_image.jpg', cropped_image)
cv2.imwrite('blue_channel_image.jpg', blue_channel_image)
cv2.imwrite('green_channel_image.jpg', green_channel_image)
cv2.imwrite('red_channel_image.jpg', red_channel_image)
cv2.imwrite('edge_image.jpg', edge_image)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()