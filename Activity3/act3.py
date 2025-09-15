import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to plot histograms
def plot_histogram(img, title, subplot_idx):
    plt.subplot(subplot_idx)
    plt.hist(img.ravel(), 256, [0, 256], color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

# Load image
img_path = 'sample.jpg'  # Replace with your image path
img = cv2.imread(img_path)

# Check if image is loaded
if img is None:
    print("Error: Could not load image. Check the file path.")
    exit()

# Part 1: Histogram Equalization
# Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply histogram equalization
equalized_img = cv2.equalizeHist(gray_img)

# For color image, convert to YUV and equalize the luminance channel
yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
yuv_img[:,:,0] = cv2.equalizeHist(yuv_img[:,:,0])
equalized_color_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)

# Part 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(gray_img)

# Display images
plt.figure(figsize=(15, 10))

# Original grayscale image and histogram
plt.subplot(231), plt.imshow(gray_img, cmap='gray'), plt.title('Original Grayscale')
plt.subplot(234), plot_histogram(gray_img, 'Original Histogram', 234)

# Equalized grayscale image and histogram
plt.subplot(232), plt.imshow(equalized_img, cmap='gray'), plt.title('Equalized Grayscale')
plt.subplot(235), plot_histogram(equalized_img, 'Equalized Histogram', 235)

# CLAHE image and histogram
plt.subplot(233), plt.imshow(clahe_img, cmap='gray'), plt.title('CLAHE Grayscale')
plt.subplot(236), plot_histogram(clahe_img, 'CLAHE Histogram', 236)

plt.tight_layout()
plt.show()

# Display color images
plt.figure(figsize=(15, 5))
plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original Color')
plt.subplot(122), plt.imshow(cv2.cvtColor(equalized_color_img, cv2.COLOR_BGR2RGB)), plt.title('Equalized Color')
plt.tight_layout()
plt.show()

# Part 2: Linear Contrast Adjustment (Contrast Stretching)
def contrast_stretching(img):
    # Get min and max pixel values
    min_val = np.min(img)
    max_val = np.max(img)
    # Apply contrast stretching
    stretched = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return stretched

stretched_img = contrast_stretching(gray_img)

# Display contrast-stretched image
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(gray_img, cmap='gray'), plt.title('Original Grayscale')
plt.subplot(122), plt.imshow(stretched_img, cmap='gray'), plt.title('Contrast Stretched')
plt.tight_layout()
plt.show()