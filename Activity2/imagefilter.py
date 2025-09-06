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

# Resize all images to 280x280 for the 3x3 grid (840/3 = 280)
display_size = (280, 280)
original_display = cv2.resize(image, display_size, interpolation=cv2.INTER_AREA)

# Convert to grayscale for edge detection
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. Identity Kernel
identity_kernel = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]], dtype=np.float32)
identity_image = cv2.filter2D(image, -1, identity_kernel)
identity_display = cv2.resize(identity_image, display_size, interpolation=cv2.INTER_AREA)

# 2. Blur Filters
# Box Blur
box_blur_image = cv2.blur(image, (5, 5))  # 5x5 kernel
box_blur_display = cv2.resize(box_blur_image, display_size, interpolation=cv2.INTER_AREA)

# Gaussian Blur
gaussian_blur_image = cv2.GaussianBlur(image, (5, 5), 0)  # 5x5 kernel, sigma=0
gaussian_blur_display = cv2.resize(gaussian_blur_image, display_size, interpolation=cv2.INTER_AREA)

# Median Blur
median_blur_image = cv2.medianBlur(image, 5)  # 5x5 kernel
median_blur_display = cv2.resize(median_blur_image, display_size, interpolation=cv2.INTER_AREA)

# 3. Sharpening Filter: Emboss
emboss_kernel = np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [ 0,  1, 2]], dtype=np.float32)
emboss_image = cv2.filter2D(image, -1, emboss_kernel)
emboss_display = cv2.resize(emboss_image, display_size, interpolation=cv2.INTER_AREA)

# 4. Edge Detection
# Sobel Edge Detection (combined X and Y directions)
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_abs_x = cv2.convertScaleAbs(sobel_x)
sobel_abs_y = cv2.convertScaleAbs(sobel_y)
sobel_image = cv2.addWeighted(sobel_abs_x, 0.5, sobel_abs_y, 0.5, 0)
sobel_image = cv2.cvtColor(sobel_image, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel
sobel_display = cv2.resize(sobel_image, display_size, interpolation=cv2.INTER_AREA)

# Canny Edge Detection
canny_image = cv2.Canny(gray_image, 100, 200)  # Thresholds 100, 200
canny_image = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel
canny_display = cv2.resize(canny_image, display_size, interpolation=cv2.INTER_AREA)

# Laplacian Edge Detection
laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
laplacian_image = cv2.convertScaleAbs(laplacian)
laplacian_image = cv2.cvtColor(laplacian_image, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel
laplacian_display = cv2.resize(laplacian_image, display_size, interpolation=cv2.INTER_AREA)

# Add labels to each image
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
color = (255, 255, 255)  # White text
thickness = 1
label_y = 20  # Position near top

cv2.putText(original_display, 'Original', (10, label_y), font, font_scale, color, thickness, cv2.LINE_AA)
cv2.putText(identity_display, 'Identity', (10, label_y), font, font_scale, color, thickness, cv2.LINE_AA)
cv2.putText(box_blur_display, 'Box Blur', (10, label_y), font, font_scale, color, thickness, cv2.LINE_AA)
cv2.putText(gaussian_blur_display, 'Gaussian Blur', (10, label_y), font, font_scale, color, thickness, cv2.LINE_AA)
cv2.putText(median_blur_display, 'Median Blur', (10, label_y), font, font_scale, color, thickness, cv2.LINE_AA)
cv2.putText(emboss_display, 'Emboss', (10, label_y), font, font_scale, color, thickness, cv2.LINE_AA)
cv2.putText(sobel_display, 'Sobel', (10, label_y), font, font_scale, color, thickness, cv2.LINE_AA)
cv2.putText(canny_display, 'Canny', (10, label_y), font, font_scale, color, thickness, cv2.LINE_AA)
cv2.putText(laplacian_display, 'Laplacian', (10, label_y), font, font_scale, color, thickness, cv2.LINE_AA)

# Create a 3x3 grid
row1 = np.hstack((original_display, identity_display, box_blur_display))
row2 = np.hstack((gaussian_blur_display, median_blur_display, emboss_display))
row3 = np.hstack((sobel_display, canny_display, laplacian_display))
panel = np.vstack((row1, row2, row3))

# Display the panel
cv2.imshow('Image Filters and Edge Detection (840x840 Input)', panel)

# Save the output images (individual)
cv2.imwrite('identity_image.jpg', identity_image)
cv2.imwrite('box_blur_image.jpg', box_blur_image)
cv2.imwrite('gaussian_blur_image.jpg', gaussian_blur_image)
cv2.imwrite('median_blur_image.jpg', median_blur_image)
cv2.imwrite('emboss_image.jpg', emboss_image)
cv2.imwrite('sobel_image.jpg', sobel_image)
cv2.imwrite('canny_image.jpg', canny_image)
cv2.imwrite('laplacian_image.jpg', laplacian_image)

# Save the panel
cv2.imwrite('filter_panel.jpg', panel)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()