import cv2
import numpy as np
from sklearn.cluster import KMeans

class ImageProcessorScript2:
    def __init__(self):
        self.segmentation_done = False
        self.segmented_labels = None

    def perform_clustering(self, image, num_clusters=4, max_iter=1000):
        try:
            data = image.reshape(-1, 3).astype(np.float64)
            kmeans = KMeans(n_clusters=num_clusters, max_iter=max_iter)
            kmeans.fit(data)
            clustered_labels = kmeans.labels_.reshape(image.shape[:2])
            self.segmentation_done = True
            self.segmented_labels = clustered_labels
            return clustered_labels
        except Exception as e:
            print(f"Error in perform_clustering: {e}")
            return None

    def enhance_follicles(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
            enhanced_image = cv2.bitwise_and(image, image, mask=binary_mask)
            enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=1.5, beta=50)
            return enhanced_image
        except Exception as e:
            print(f"Error in enhance_follicles: {e}")
            return None

    def process_image(self, image_path):
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Error: Unable to read the image '{image_path}'.")
                return None, None, None, None, None, None, None, None

            resized_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

            if not self.segmentation_done:
                enhanced_image = self.enhance_follicles(resized_image)
                if enhanced_image is None:
                    return None, None, None, None, None, None, None, None

                gray_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
                segmented_labels = self.perform_clustering(enhanced_image, max_iter=500)
                if segmented_labels is None:
                    return None, None, None, None, None, None, None, None
            else:
                enhanced_image = None
                gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                segmented_labels = self.segmented_labels

            edges_canny = cv2.Canny(gray_image, 50, 150)

            height, width = edges_canny.shape
            crop_x = 0
            crop_y = height // 4
            crop_height = height // 2
            crop_width = width
            cropped_image_canny = edges_canny[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

            contours_canny, _ = cv2.findContours(cropped_image_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours based on area
            min_area = 1
            max_area = 999999
            large_contours = [cnt for cnt in contours_canny if min_area < cv2.contourArea(cnt) < max_area]

            num_follicles_canny = len(large_contours)

            return resized_image, enhanced_image, gray_image, segmented_labels, edges_canny, cropped_image_canny, large_contours, num_follicles_canny
        except Exception as e:
            print(f"Error processing image in Script2: {e}")
            return None, None, None, None, None, None, None, None
