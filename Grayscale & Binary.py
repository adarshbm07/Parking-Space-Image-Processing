import cv2
import os

# Assuming this script runs from inside E:\Parking_space
input_folder = "folder1_input_images"
output_base = "folder2_gray_bw"
grayscale_folder = os.path.join(output_base, "grayscale")
binary_folder = os.path.join(output_base, "binary")

# Create subfolders
os.makedirs(grayscale_folder, exist_ok=True)
os.makedirs(binary_folder, exist_ok=True)

# Process each image
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        input_path = os.path.join(input_folder, filename)
        img = cv2.imread(input_path)

        if img is None:
            print(f"❌ Could not read image: {filename}")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_path = os.path.join(grayscale_folder, f"gray_{filename}")
        cv2.imwrite(gray_path, gray)

        # Convert to binary using thresholding
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        binary_path = os.path.join(binary_folder, f"binary_{filename}")
        cv2.imwrite(binary_path, binary)

        print(f"✅ Processed: {filename}")

print("\n🎉 All grayscale and binary images saved in their respective folders.")
