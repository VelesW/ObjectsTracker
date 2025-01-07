import numpy as np
import cv2

class Histogram:
    def __init__(self, matrix, w, h):
        self.N = w * h
        self.number = self.generate_histogram(matrix, w, h)
        self.intensity = self.calc_total_intensity(self.number)
        self.treshold = self.get_treshold(self.number, self.intensity, self.N)

    @property
    def Treshold(self):
        return self.treshold * 2

    def generate_histogram(self, matrix, w, h):
        histogram = [0] * 256
        for x in range(w):
            for y in range(h):
                v = matrix[x, y]
                histogram[v] += 1
        return histogram

    def calc_total_intensity(self, histogram):
        result = 0
        for i in range(len(histogram)):
            result += i * histogram[i]
        return result

    def get_treshold(self, histogram, total_intensity, total_pixels):
        result = 0
        variance = 0
        best_var = float('-inf')
        mean_bg = 0
        weight_bg = 0
        mean_fg = total_intensity / total_pixels
        weight_fg = self.N
        current_treshold = 0

        while current_treshold < 255:
            diff_means = mean_fg - mean_bg
            variance = weight_bg * weight_fg * diff_means * diff_means

            if variance > best_var:
                best_var = variance
                result = current_treshold

            while current_treshold < 255 and histogram[current_treshold] == 0:
                current_treshold += 1

            if current_treshold >= 255:
                break

            weight_bg += histogram[current_treshold]
            weight_fg -= histogram[current_treshold]

            mean_bg = ((mean_bg * weight_bg) + (histogram[current_treshold] * current_treshold)) / weight_bg
            mean_fg = ((mean_fg * weight_fg) - (histogram[current_treshold] * current_treshold)) / weight_fg

            current_treshold += 1

        return result
    
    def apply_threshold(self, image):
        _, binary_image = cv2.threshold(image, self.treshold, 255, cv2.THRESH_BINARY)

        # Morphological opening to clean up the binary image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

        return cleaned_image
