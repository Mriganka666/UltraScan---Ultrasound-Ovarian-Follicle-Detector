import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from PIL import Image as PILImage
import cv2
import numpy as np

class ImageAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Area of Contours")

        self.label_dpi = tk.Label(root, text="DPI: ")
        self.label_dpi.pack()

        self.label_resolution = tk.Label(root, text="Resolution: ")
        self.label_resolution.pack()

        self.label_filepath = tk.Label(root, text="File Path: ")
        self.label_filepath.pack()

        self.text_filepath = tk.Label(root, text="", wraplength=400, anchor="w")
        self.text_filepath.pack()

        self.btn_upload = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.btn_upload.pack(pady=10)

        self.btn_show_areas_mm = tk.Button(root, text="Contour Sizes (MM²)", command=self.show_areas_in_mm)
        self.btn_show_areas_mm.pack(pady=10)

        self.btn_clear = tk.Button(root, text="Clear", command=self.clear_all)
        self.btn_clear.pack(pady=10)

        self.scrollbar = tk.Scrollbar(root)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, yscrollcommand=self.scrollbar.set, width=40, height=10)
        self.text_area.pack(padx=10, pady=10)

        self.scrollbar.config(command=self.text_area.yview)

        self.contour_sizes = []
        self.dpi = 96  # Default DPI value

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
        if file_path:
            try:
                self.calculate_dpi_and_resolution(file_path)
                self.calculate_contour_sizes(file_path)
                self.display_filepath(file_path)
            except Exception as e:
                self.text_area.configure(state='normal')
                self.text_area.delete('1.0', tk.END)
                self.text_area.insert(tk.END, f"Error: {str(e)}")
                self.text_area.configure(state='disabled')
                messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def calculate_dpi_and_resolution(self, file_path):
        img = PILImage.open(file_path)
        self.dpi = img.info.get('dpi', (96, 96))[0]  # Default to 96 if DPI not found
        width, height = img.size
        self.label_dpi.config(text=f"DPI: {self.dpi}")
        self.label_resolution.config(text=f"Resolution: {width} x {height}")

    def calculate_contour_sizes(self, file_path):
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        self.contour_sizes = []
        for c in contours:
            area = cv2.contourArea(c)
            # Convert area to square millimeters
            px_to_mm = 25.4 / self.dpi
            area_mm2 = area * (px_to_mm ** 2)
            self.contour_sizes.append(area_mm2)
        self.contour_sizes.sort()  # Sort sizes in ascending order

    def show_areas_in_mm(self):
        if not self.contour_sizes:
            sizes_text = "No contours found."
        else:
            sizes_text = ", ".join(f"{size:.2f}" for size in self.contour_sizes)
        
        self.text_area.configure(state='normal')
        self.text_area.delete('1.0', tk.END)
        self.text_area.insert(tk.END, sizes_text)
        self.text_area.configure(state='disabled')

    def display_filepath(self, file_path):
        self.text_filepath.config(text=f"File Path: {file_path}")

    def clear_all(self):
        self.text_filepath.config(text="File Path: ")
        self.label_dpi.config(text="DPI: ")
        self.label_resolution.config(text="Resolution: ")
        self.text_area.configure(state='normal')
        self.text_area.delete('1.0', tk.END)
        self.text_area.configure(state='disabled')
        self.contour_sizes = []
        self.dpi = 96  # Reset to default DPI

root = tk.Tk()
app = ImageAnalyzer(root)
root.mainloop()
