import cv2
import os
import numpy as np

# Input from noise filtered folder (or use from histogram equalized)
input_folder = "folder3_hist_eq"
output_base = "folder6_image_enhancement"

# Output subfolders
clahe_folder = os.path.join(output_base, "clahe")
sharpened_folder = os.path.join(output_base, "sharpened")

# Create output folders
os.makedirs(clahe_folder, exist_ok=True)
os.makedirs(sharpened_folder, exist_ok=True)

# Create CLAHE object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Sharpening kernel
sharpening_kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])

# Process images
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        input_path = os.path.join(input_folder, filename)
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"❌ Could not read image: {filename}")
            continue

        # 1. CLAHE Enhancement
        clahe_img = clahe.apply(img)
        cv2.imwrite(os.path.join(clahe_folder, f"clahe_{filename}"), clahe_img)

        # 2. Image Sharpening
        sharpened_img = cv2.filter2D(img, -1, sharpening_kernel)
        cv2.imwrite(os.path.join(sharpened_folder, f"sharpened_{filename}"), sharpened_img)

        print(f"✅ Enhanced: {filename}")

print("\n🎉 Image enhancement complete. Results saved in folder6_image_enhancement.")
