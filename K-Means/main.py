import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import cv2
import time
import matplotlib.gridspec as gridspec
from threading import Thread
from script1 import ImageProcessorScript1
from script2 import ImageProcessorScript2

class ImageProcessingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("UltraScan Using K-Means")

        self.image_path = ""
        self.processor_script1 = ImageProcessorScript1()
        self.processor_script2 = ImageProcessorScript2()
        self.segmentation_done = False
        self.create_widgets()

    def create_widgets(self):
        self.choose_button = tk.Button(self.master, text="Choose Image", command=self.choose_image)
        self.choose_button.pack(pady=10)

        self.image_selected_label = tk.Label(self.master, text="")
        self.image_selected_label.pack(pady=5)

        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack(pady=10)

        self.script1_button = tk.Button(self.button_frame, text="Segmentation", command=self.run_script1)
        self.script1_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.timer_label1 = tk.Label(self.button_frame, text="Execution Time: -")
        self.timer_label1.pack(side=tk.LEFT, padx=5, pady=5)

        self.script2_button = tk.Button(self.button_frame, text="Detection", command=self.run_script2)
        self.script2_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.timer_label2 = tk.Label(self.button_frame, text="Execution Time: -")
        self.timer_label2.pack(side=tk.LEFT, padx=5, pady=5)

    def choose_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            self.image_selected_label.config(text=f"Selected image: {self.image_path}")
        else:
            messagebox.showwarning("No Image Selected", "Please select an image.")

    def run_script1(self):
        if not self.image_path:
            messagebox.showwarning("No Image Selected", "Please select an image first.")
            return

        self.script1_button.config(state=tk.DISABLED)
        start_time = time.time()

        thread = Thread(target=self.run_script1_thread, args=(start_time,))
        thread.start()

    def run_script1_thread(self, start_time):
        resized_image, enhanced_image, segmented_labels, silhouette, davies_bouldin = self.processor_script1.process_image(self.image_path)
        end_time = time.time()
        elapsed_time = end_time - start_time

        self.master.after(0, self.update_timer_label1, elapsed_time)

        if resized_image is not None:
            self.show_image_script1(resized_image, segmented_labels, silhouette, davies_bouldin)
            self.segmentation_done = True
        self.script1_button.config(state=tk.NORMAL)

    def update_timer_label1(self, elapsed_time):
        self.timer_label1.config(text=f"Execution Time: {elapsed_time:.2f} seconds ✓")

    def show_image_script1(self, resized_image, segmented_labels, silhouette, davies_bouldin):
        self.clear_canvas()

        fig = plt.figure(figsize=(18, 6), facecolor='white')
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

        ax_original = fig.add_subplot(gs[0, 0])
        ax_original.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        ax_original.set_title('Original Image', color='black')
        ax_original.axis('off')

        ax_segmented = fig.add_subplot(gs[0, 1])
        ax_segmented.imshow(segmented_labels, cmap='gray')
        ax_segmented.set_title('Segmented Image', color='black')
        ax_segmented.axis('off')

        ax_metrics = fig.add_subplot(gs[0, 2])
        ax_metrics.text(0.1, 0.5, f"Silhouette Score: {silhouette:.2f}\nDavies-Bouldin Index: {davies_bouldin:.2f}", 
                        fontsize=12, ha='left', va='center', color='black')
        ax_metrics.set_axis_off()
        ax_metrics.set_title('Clustering Metrics', color='black')

        self.display_canvas(fig)

    def run_script2(self):
        if not self.image_path:
            messagebox.showwarning("No Image Selected", "Please select an image first.")
            return

 

        self.script2_button.config(state=tk.DISABLED)
        start_time = time.time()

        thread = Thread(target=self.run_script2_thread, args=(start_time,))
        thread.start()

    def run_script2_thread(self, start_time):
        resized_image, enhanced_image, gray_image, segmented_labels, edges_canny, cropped_image_canny, contours_canny, num_follicles_canny = self.processor_script2.process_image(self.image_path)
        end_time = time.time()
        elapsed_time = end_time - start_time

        self.master.after(0, self.update_timer_label2, elapsed_time)

        if resized_image is not None:
            self.show_image_script2(resized_image, enhanced_image, gray_image, segmented_labels, edges_canny, cropped_image_canny, contours_canny, num_follicles_canny)
        self.script2_button.config(state=tk.NORMAL)

    def update_timer_label2(self, elapsed_time):
        self.timer_label2.config(text=f"Execution Time: {elapsed_time:.2f} seconds ✓")

    def show_image_script2(self, resized_image, enhanced_image, gray_image, segmented_labels, edges_canny, cropped_image_canny, contours_canny, num_follicles_canny):
        self.clear_canvas()

        fig = plt.figure(figsize=(18, 12), facecolor='white')
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])

        ax_original = fig.add_subplot(gs[0, 0])
        ax_original.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        ax_original.set_title('Original Image', color='black')
        ax_original.axis('off')

        ax_enhanced = fig.add_subplot(gs[0, 1])
        ax_enhanced.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
        ax_enhanced.set_title('Enhanced Image', color='black')
        ax_enhanced.axis('off')

        ax_gray = fig.add_subplot(gs[0, 2])
        ax_gray.imshow(gray_image, cmap='gray')
        ax_gray.set_title('Grayscale Image', color='black')
        ax_gray.axis('off')

        ax_segmented = fig.add_subplot(gs[1, 0])
        ax_segmented.imshow(segmented_labels, cmap='gray')
        ax_segmented.set_title('Segmented Image', color='black')
        ax_segmented.axis('off')

        ax_edges_canny = fig.add_subplot(gs[1, 1])
        ax_edges_canny.imshow(edges_canny, cmap='gray')
        ax_edges_canny.set_title('Edge Detection', color='black')
        ax_edges_canny.axis('off')

        ax_contours_canny = fig.add_subplot(gs[1, 2])
        cropped_color = cv2.cvtColor(cropped_image_canny, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(cropped_color, contours_canny, -1, (0, 0, 255), 1)
        cropped_color_rgb = cv2.cvtColor(cropped_color, cv2.COLOR_BGR2RGB)
        ax_contours_canny.imshow(cropped_color_rgb)
        ax_contours_canny.set_title(f'Follicle Detection: {num_follicles_canny} detected', color='black')
        ax_contours_canny.axis('off')

        self.display_canvas(fig)



    def display_canvas(self, fig):
        canvas = FigureCanvasTkAgg(fig, master=self.master)
        canvas.draw()
        canvas.get_tk_widget().pack(padx=10, pady=10)
        self.current_canvas = canvas

    def clear_canvas(self):
        if hasattr(self, 'current_canvas'):
            self.current_canvas.get_tk_widget().pack_forget() # type: ignore
            self.current_canvas = None

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
