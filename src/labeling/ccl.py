import numpy as np
import cv2
from typing import List, Tuple

class ConnectedComponentLabeling:
    def __init__(self, binary_image: np.ndarray):
        # Initialize the class with a binary image (e.g., after thresholding).
        # The image should contain binary values (0 or 255).
        self.binary_image = binary_image
        self.labels = None  # Placeholder for the labeled image.
        self.num_labels = 0  # Counter for the number of connected components.

    def label_components(self) -> Tuple[np.ndarray, int]:
        # Perform connected component labeling on the binary image.
        # Uses OpenCV's `connectedComponents` function with 8-connectivity.
        # Returns a labeled image and the total number of labels.
        self.num_labels, self.labels = cv2.connectedComponents(self.binary_image, connectivity=8)
        return self.labels, self.num_labels

    def get_blobs(self) -> List[np.ndarray]:
        # Extract individual blobs (connected components) as separate regions.
        # Each blob is represented as a binary mask (array of 0s and 255s).
        blobs = []
        for label in range(1, self.num_labels):  # Skip label 0 (background).
            blobs.append((self.labels == label).astype(np.uint8) * 255)
        return blobs
    
    def count_blobs(self) -> int:
        """Count the number of blobs in the labeled image."""
        # If the image hasn't been labeled yet, return 0.
        if self.labels is None:
            return 0

        # Count unique labels, excluding the background label (0).
        unique_labels = np.unique(self.labels)
        num_blobs = len(unique_labels) - 1  # Exclude background label 0
        return num_blobs
