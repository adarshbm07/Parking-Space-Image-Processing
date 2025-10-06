import cv2
import os

# Input from histogram equalized images
input_folder = "folder3_hist_eq"
output_base = "folder4_spatial_filter"

# Output subfolders
gaussian_folder = os.path.join(output_base, "gaussian")
median_folder = os.path.join(output_base, "median")
bilateral_folder = os.path.join(output_base, "bilateral")

# Create output folders
os.makedirs(gaussian_folder, exist_ok=True)
os.makedirs(median_folder, exist_ok=True)
os.makedirs(bilateral_folder, exist_ok=True)

# Process each image
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        input_path = os.path.join(input_folder, filename)
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"❌ Could not read image: {filename}")
            continue

        # Gaussian Blur
        gaussian = cv2.GaussianBlur(img, (5, 5), 0)
        cv2.imwrite(os.path.join(gaussian_folder, f"gaussian_{filename}"), gaussian)

        # Median Blur
        median = cv2.medianBlur(img, 5)
        cv2.imwrite(os.path.join(median_folder, f"median_{filename}"), median)

        # Bilateral Filter
        bilateral = cv2.bilateralFilter(img, 9, 75, 75)
        cv2.imwrite(os.path.join(bilateral_folder, f"bilateral_{filename}"), bilateral)

        print(f"✅ Filtered: {filename}")

print("\n🎉 All filtered images saved in folder4_spatial_filter.")
