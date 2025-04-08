import cv2
import numpy as np
import skfuzzy as fuzz
from sklearn.metrics import silhouette_score, davies_bouldin_score

class ImageProcessorScript1:
    def __init__(self):
        pass

    def perform_clustering(self, image, num_clusters=4, fuzziness=2, error=0.005, max_iter=1000):
        try:
            data = image.reshape(-1, 3).astype(np.float64)
            _, u, _, _, _, _, _ = fuzz.cmeans(data.T, c=num_clusters, m=fuzziness, error=error, maxiter=max_iter, init=None)
            clustered_labels = np.argmax(u, axis=0).reshape(image.shape[:2])
            return clustered_labels
        except Exception as e:
            print(f"Error in perform_clustering: {e}")
            return None

    def enhance_image(self, image):
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Enhance image
            enhanced_image = cv2.convertScaleAbs(gray, alpha=1.5, beta=50)
            return enhanced_image
        except Exception as e:
            print(f"Error in enhance_image: {e}")
            return None

    def process_image(self, image_path):
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Error: Unable to read the image '{image_path}'.")
                return None, None, None, None, None

            # Resize image
            resized_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

            # Enhance image
            enhanced_image = self.enhance_image(resized_image)
            if enhanced_image is None:
                return None, None, None, None, None

            # Perform clustering
            segmented_labels = self.perform_clustering(resized_image, max_iter=500)
            if segmented_labels is None:
                return None, None, None, None, None

            # Flatten the image and labels for clustering metrics
            flat_image = resized_image.reshape(-1, 3).astype(np.float64)
            flat_labels = segmented_labels.flatten()

            # Calculate silhouette score
            silhouette = silhouette_score(flat_image, flat_labels)

            # Calculate Davies-Bouldin score
            davies_bouldin = davies_bouldin_score(flat_image, flat_labels)

            return resized_image, enhanced_image, segmented_labels, silhouette, davies_bouldin
        except Exception as e:
            print(f"Error processing image in Script1: {e}")
            return None, None, None, None, None
