import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk

class ParkingSpaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚗 Parking Lot - Car and Empty Slot Detection")

        # Frame for buttons
        self.frame = tk.Frame(root)
        self.frame.pack(padx=10, pady=10)

        self.open_button = tk.Button(self.frame, text="📂 Open Image", command=self.open_image, width=15)
        self.open_button.grid(row=0, column=0, padx=5, pady=5)

        self.process_button = tk.Button(self.frame, text="⚙️ Process Image", command=self.process_image, width=15)
        self.process_button.grid(row=0, column=1, padx=5, pady=5)

        # Canvas for image display
        self.canvas = tk.Canvas(root, width=800, height=600, bg="gray")
        self.canvas.pack()

        # Status label
        self.status_label = tk.Label(root, text="Please load an image to begin.", font=("Helvetica", 12))
        self.status_label.pack(pady=5)

        self.original_image = None
        self.processed_image = None
        self.image_tk = None  # To prevent garbage collection

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                self.display_image(self.original_image)
                self.status_label.config(text="✅ Image loaded. Click 'Process Image' to detect.")
            else:
                self.status_label.config(text="❌ Error loading image.")

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((800, 600))  # Resize for consistent display
        self.image_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)

    def process_image(self):
        if self.original_image is None:
            self.status_label.config(text="⚠️ No image loaded.")
            return

        img = self.original_image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1)

        # Adaptive thresholding for lighting variation
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 10)

        kernel = np.ones((5, 5), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        morph = cv2.dilate(morph, kernel, iterations=1)

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"[INFO] Total contours detected: {len(contours)}")

        output = img.copy()
        car_count = 0
        empty_count = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500 or area > 60000:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h != 0 else 0
            if not (0.3 < aspect_ratio < 6.5):
                continue

            roi = gray[y:y + h, x:x + w]
            std_dev = np.std(roi)
            mean_intensity = np.mean(roi)
            edges = cv2.Canny(roi, 40, 100)
            edge_count = np.sum(edges > 0)

            label = f"std:{std_dev:.1f} mean:{mean_intensity:.1f} edge#: {edge_count}"
            cv2.putText(output, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

            is_car = std_dev > 5 and edge_count > 180

            if is_car:
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green = Car
                car_count += 1
            else:
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red = Empty
                empty_count += 1

        self.processed_image = output
        self.display_image(self.processed_image)
        result_msg = f"✅ Detection complete - Cars: {car_count} | Empty Slots: {empty_count}"
        self.status_label.config(text=result_msg)
        print(result_msg)


if __name__ == "__main__":
    root = tk.Tk()
    app = ParkingSpaceRecognitionApp(root)
    root.mainloop()
