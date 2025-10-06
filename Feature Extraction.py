import cv2
import os
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt

# Input folder from CLAHE-enhanced images
input_folder = "folder6_image_enhancement/clahe"
output_base = "folder7_feature_extraction"

# Output folders
hog_folder = os.path.join(output_base, "hog_features")
hist_folder = os.path.join(output_base, "histogram_features")

os.makedirs(hog_folder, exist_ok=True)
os.makedirs(hist_folder, exist_ok=True)

# HOG parameters
hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'visualize': True,
    'transform_sqrt': True
}

# Process images
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        input_path = os.path.join(input_folder, filename)
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"❌ Could not read image: {filename}")
            continue

        # HOG feature extraction
        hog_features, hog_image = hog(img, **hog_params)
        hog_img_path = os.path.join(hog_folder, f"hog_{filename}")
        plt.imsave(hog_img_path, hog_image, cmap='gray')

        # Histogram feature extraction
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = hist.flatten()
        hist_img = np.full((300, 256), 255, dtype=np.uint8)
        cv2.normalize(hist, hist, 0, 300, cv2.NORM_MINMAX)

        for x, y in enumerate(hist):
            cv2.line(hist_img, (x, 300), (x, 300 - int(y)), 0)

        hist_img_path = os.path.join(hist_folder, f"hist_{filename}")
        cv2.imwrite(hist_img_path, hist_img)

        print(f"✅ Features extracted: {filename}")

print("\n🎉 Feature extraction complete. Saved in folder7_feature_extraction.")
