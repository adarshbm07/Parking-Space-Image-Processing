import cv2
import os
import numpy as np

# Input from histogram equalized images
input_folder = "folder3_hist_eq"
output_base = "folder5_noise_filtering"

# Create output folders
noisy_folder = os.path.join(output_base, "noisy")
gaussian_denoised_folder = os.path.join(output_base, "gaussian_denoised")
median_denoised_folder = os.path.join(output_base, "median_denoised")
bilateral_denoised_folder = os.path.join(output_base, "bilateral_denoised")

os.makedirs(noisy_folder, exist_ok=True)
os.makedirs(gaussian_denoised_folder, exist_ok=True)
os.makedirs(median_denoised_folder, exist_ok=True)
os.makedirs(bilateral_denoised_folder, exist_ok=True)

# Function to add Gaussian noise
def add_gaussian_noise(image):
    mean = 0
    var = 20
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy = cv2.add(image, gaussian)
    return noisy

# Process images
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        input_path = os.path.join(input_folder, filename)
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"❌ Could not read image: {filename}")
            continue

        # Step 1: Add Gaussian noise
        noisy_img = add_gaussian_noise(img)
        noisy_path = os.path.join(noisy_folder, f"noisy_{filename}")
        cv2.imwrite(noisy_path, noisy_img)

        # Step 2: Apply Gaussian Blur
        gaussian_denoised = cv2.GaussianBlur(noisy_img, (5, 5), 0)
        cv2.imwrite(os.path.join(gaussian_denoised_folder, f"gaussian_denoised_{filename}"), gaussian_denoised)

        # Step 3: Apply Median Filter
        median_denoised = cv2.medianBlur(noisy_img, 5)
        cv2.imwrite(os.path.join(median_denoised_folder, f"median_denoised_{filename}"), median_denoised)

        # Step 4: Apply Bilateral Filter
        bilateral_denoised = cv2.bilateralFilter(noisy_img, 9, 75, 75)
        cv2.imwrite(os.path.join(bilateral_denoised_folder, f"bilateral_denoised_{filename}"), bilateral_denoised)

        print(f"✅ Noise added and removed: {filename}")

print("\n🎉 All noisy and denoised images saved in folder5_noise_filtering.")
