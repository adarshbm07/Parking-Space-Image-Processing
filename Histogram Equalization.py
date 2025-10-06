import cv2
import os

# Input grayscale folder from previous step
input_folder = "folder2_gray_bw/grayscale"
output_folder = "folder3_hist_eq"

# Create output directory
os.makedirs(output_folder, exist_ok=True)

# Process each grayscale image
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        input_path = os.path.join(input_folder, filename)
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"❌ Could not read image: {filename}")
            continue

        # Apply histogram equalization
        hist_eq = cv2.equalizeHist(img)

        # Save result
        output_path = os.path.join(output_folder, f"hist_eq_{filename}")
        cv2.imwrite(output_path, hist_eq)
        print(f"✅ Histogram Equalized: {filename}")

print("\n🎉 All images saved in folder3_hist_eq.")
