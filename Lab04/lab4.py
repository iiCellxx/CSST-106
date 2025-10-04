import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# --- Part 1: ORB and SIFT Feature Matching Between Two Images ---
def orb_feature_matching(img1_path, img2_path, nfeatures=1000):
    """Perform ORB and SIFT feature detection and matching, return results for panel."""
    # Load images in grayscale
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print('Error loading one or both images')
        return None, None, None, None

    # Initialize data for chart
    labels = ['ORB Img1', 'ORB Img2', 'ORB Matches', 'SIFT Img1', 'SIFT Img2', 'SIFT Matches']
    values = [0] * 6
    colors = ['#36A2EB', '#4CAF50', '#FF9800', '#AB47BC', '#EC407A', '#FFEB3B']  # ORB: blue, green, orange; SIFT: purple, magenta, yellow
    sift_available = True

    # --- ORB Matching ---
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints1_orb, descriptors1_orb = orb.detectAndCompute(img1, None)
    keypoints2_orb, descriptors2_orb = orb.detectAndCompute(img2, None)

    print(f"ORB Matching - Keypoints in image 1: {len(keypoints1_orb)}")
    print(f"ORB Matching - Keypoints in image 2: {len(keypoints2_orb)}")

    bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_orb = bf_orb.knnMatch(descriptors1_orb, descriptors2_orb, k=2)

    good_matches_orb = []
    for m, n in matches_orb:
        if m.distance < 0.7 * n.distance:
            good_matches_orb.append(m)

    print(f"ORB Matching - Number of good matches: {len(good_matches_orb)}")

    img_matches_orb = cv2.drawMatches(img1, keypoints1_orb, img2, keypoints2_orb, good_matches_orb, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches_orb = cv2.cvtColor(img_matches_orb, cv2.COLOR_BGR2RGB)  # Convert for matplotlib

    values[0] = len(keypoints1_orb)
    values[1] = len(keypoints2_orb)
    values[2] = len(good_matches_orb)

    # --- SIFT Matching ---
    img_matches_sift = None
    try:
        sift = cv2.SIFT_create()
        keypoints1_sift, descriptors1_sift = sift.detectAndCompute(img1, None)
        keypoints2_sift, descriptors2_sift = sift.detectAndCompute(img2, None)

        print(f"SIFT Matching - Keypoints in image 1: {len(keypoints1_sift)}")
        print(f"SIFT Matching - Keypoints in image 2: {len(keypoints2_sift)}")

        bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches_sift = bf_sift.knnMatch(descriptors1_sift, descriptors2_sift, k=2)

        good_matches_sift = []
        for m, n in matches_sift:
            if m.distance < 0.7 * n.distance:
                good_matches_sift.append(m)

        print(f"SIFT Matching - Number of good matches: {len(good_matches_sift)}")

        img_matches_sift = cv2.drawMatches(img1, keypoints1_sift, img2, keypoints2_sift, good_matches_sift, None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        img_matches_sift = cv2.cvtColor(img_matches_sift, cv2.COLOR_BGR2RGB)

        values[3] = len(keypoints1_sift)
        values[4] = len(keypoints2_sift)
        values[5] = len(good_matches_sift)
    except cv2.error:
        print("SIFT not available in this OpenCV configuration. Skipping SIFT.")
        sift_available = False
        keypoints1_sift, descriptors1_sift, keypoints2_sift, descriptors2_sift = [], None, [], None
        good_matches_sift = []

    return img_matches_orb, img_matches_sift, labels, values

# --- Part 2: Compare ORB and SIFT ---
def compare_detectors(img_path, nfeatures=1000):
    """Compare ORB and SIFT, return keypoint images and chart data for panel."""
    # Load image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('Error loading image')
        return None, None, None, None, None

    # Lists for chart data
    detector_names = ['ORB', 'SIFT']
    keypoint_counts = [0, 0]
    processing_times = [0, 0]
    sift_available = True

    # Function to detect keypoints/descriptors and measure time
    def detect_features(detector, img, detector_name):
        start_time = time.time()
        keypoints, descriptors = detector.detectAndCompute(img, None)
        end_time = time.time()
        print(f"{detector_name} - Keypoints: {len(keypoints)}, Time: {end_time - start_time:.4f} seconds")
        return keypoints, descriptors, end_time - start_time

    # ORB
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints_orb, descriptors_orb, time_orb = detect_features(orb, img, "ORB")
    img_orb = cv2.drawKeypoints(img, keypoints_orb, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_orb = cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB)
    keypoint_counts[0] = len(keypoints_orb)
    processing_times[0] = time_orb

    # SIFT
    img_sift = img.copy()
    try:
        sift = cv2.SIFT_create()
        keypoints_sift, descriptors_sift, time_sift = detect_features(sift, img, "SIFT")
        img_sift = cv2.drawKeypoints(img, keypoints_sift, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_sift = cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB)
        keypoint_counts[1] = len(keypoints_sift)
        processing_times[1] = time_sift
    except cv2.error:
        print("SIFT not available in this OpenCV configuration. Skipping SIFT.")
        sift_available = False
        keypoints_sift, descriptors_sift = [], None

    return img_orb, img_sift, detector_names, keypoint_counts, processing_times

# --- Part 3: ORB without Orientation (Emulating U-SURF) ---
def orb_no_orientation(img_path, nfeatures=1000):
    """Perform ORB feature detection without orientation, return image for panel."""
    # Load image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('Error loading image')
        return None

    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=nfeatures, scoreType=cv2.ORB_FAST_SCORE)
    keypoints, descriptors = orb.detectAndCompute(img, None)

    print(f"ORB (No Orientation) - Keypoints detected: {len(keypoints)}")

    img_keypoints = cv2.drawKeypoints(img, keypoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_keypoints = cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB)

    return img_keypoints

# --- Main Execution ---
if __name__ == "__main__":
    # File paths for images (replace with actual image paths)
    image1_path = 'image1.jpg'
    image2_path = 'image2.jpg'
    sample_image_path = 'sample_image.jpg'

    # Create a single figure with subplots
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('All Results: ORB and SIFT Feature Detection and Matching', fontsize=16)

    # Part 1: ORB and SIFT Feature Matching
    print("Running ORB and SIFT Feature Matching...")
    img_matches_orb, img_matches_sift, match_labels, match_values = orb_feature_matching(image1_path, image2_path)

    # Part 2: Compare ORB and SIFT
    print("\nRunning Comparison of ORB and SIFT...")
    img_orb, img_sift, detector_names, keypoint_counts, processing_times = compare_detectors(sample_image_path)

    # Part 3: ORB without Orientation
    print("\nRunning ORB (No Orientation) Detection...")
    img_no_orientation = orb_no_orientation(sample_image_path)

    # Plot all results in subplots
    # Row 1: ORB Matches, SIFT Matches, Matching Chart
    if img_matches_orb is not None:
        plt.subplot(3, 3, 1)
        plt.imshow(img_matches_orb)
        plt.title('ORB Matches')
        plt.axis('off')
    if img_matches_sift is not None:
        plt.subplot(3, 3, 2)
        plt.imshow(img_matches_sift)
        plt.title('SIFT Matches')
        plt.axis('off')
    if match_values:
        plt.subplot(3, 3, 3)
        plt.bar(match_labels, match_values, color=['#36A2EB', '#4CAF50', '#FF9800', '#AB47BC', '#EC407A', '#FFEB3B'])
        plt.title('Matching Results')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, max(match_values) * 1.1)
        plt.tight_layout()

    # Row 2: ORB Keypoints, SIFT Keypoints, Keypoint Count Chart
    if img_orb is not None:
        plt.subplot(3, 3, 4)
        plt.imshow(img_orb)
        plt.title('ORB Keypoints')
        plt.axis('off')
    if img_sift is not None:
        plt.subplot(3, 3, 5)
        plt.imshow(img_sift)
        plt.title('SIFT Keypoints')
        plt.axis('off')
    if keypoint_counts:
        plt.subplot(3, 3, 6)
        plt.bar(detector_names, keypoint_counts, color=['#36A2EB', '#4CAF50'])
        plt.title('Keypoint Counts')
        plt.xlabel('Detector')
        plt.ylabel('Number of Keypoints')
        plt.ylim(0, max(keypoint_counts) * 1.1)

    # Row 3: ORB No Orientation, Empty, Processing Time Chart
    if img_no_orientation is not None:
        plt.subplot(3, 3, 7)
        plt.imshow(img_no_orientation)
        plt.title('ORB (No Orientation) Keypoints')
        plt.axis('off')
    if processing_times:
        plt.subplot(3, 3, 9)
        plt.bar(detector_names, processing_times, color=['#36A2EB', '#4CAF50'])
        plt.title('Processing Times')
        plt.xlabel('Detector')
        plt.ylabel('Time (seconds)')
        plt.ylim(0, max(processing_times) * 1.1)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('all_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Close any OpenCV windows (though none are opened)
    cv2.destroyAllWindows()