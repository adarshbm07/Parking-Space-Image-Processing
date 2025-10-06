import cv2
import os
import numpy as np
from skimage.feature import hog
import pandas as pd

# Input folder (enhanced CLAHE images for fresh feature extraction)
input_folder = "folder6_image_enhancement/clahe"
output_folder = "folder8_display_features"

os.makedirs(output_folder, exist_ok=True)

# HOG parameters
hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'visualize': False,
    'transform_sqrt': True
}

hog_features_list = []
histogram_features_list = []
file_names = []

# Process each image
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        file_path = os.path.join(input_folder, filename)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # HOG feature extraction (no image returned)
        hog_feature = hog(img, **hog_params)
        hog_features_list.append(hog_feature)

        # Histogram feature (256 bins)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histogram_features_list.append(hist)

        file_names.append(filename)

# Convert to NumPy arrays
hog_features_np = np.array(hog_features_list)
histogram_features_np = np.array(histogram_features_list)

# Combine both
combined_features = np.hstack((hog_features_np, histogram_features_np))

# Save to NPZ
np.savez(os.path.join(output_folder, "features.npz"),
         filenames=file_names,
         hog=hog_features_np,
         histogram=histogram_features_np,
         combined=combined_features)

# Save to CSV
df = pd.DataFrame(combined_features)
df.insert(0, "filename", file_names)
df.to_csv(os.path.join(output_folder, "features.csv"), index=False)

print("\n🎉 Features saved as 'features.npz' and 'features.csv' in folder8_display_features.")
