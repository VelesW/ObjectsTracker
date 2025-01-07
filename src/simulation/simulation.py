import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import numpy as np
from data_generation import DataGenerator
from otsu_method import Histogram
from labeling import ConnectedComponentLabeling
import threading

class RadarSimulation:
    def __init__(self):
        self.generator = DataGenerator(frame_shape=(600, 800), max_objects=10)
        self.frames = self.generator.generate_frames(num_frames=150)
        self.current_frame_idx = 0
        self.animation_running = True
        self.otsu_applied = False

        # Prepare the figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title("Radar Simulation")
        self.image = self.ax.imshow(self.frames[0], cmap='gray')
        self.ax.axis('off')

        # Add buttons
        self.resume_button_ax = self.fig.add_axes([0.1, 0.02, 0.1, 0.05])
        self.otsu_button_ax = self.fig.add_axes([0.22, 0.02, 0.1, 0.05])
        self.label_button_ax = self.fig.add_axes([0.34, 0.02, 0.1, 0.05])

        self.resume_button = Button(self.resume_button_ax, 'New Data')
        self.otsu_button = Button(self.otsu_button_ax, 'Apply Otsu')
        self.label_button = Button(self.label_button_ax, 'Labeling')

        self.resume_button.on_clicked(self.resume_simulation)
        self.otsu_button.on_clicked(self.apply_otsu)
        self.label_button.on_clicked(self.count_blobs_in_otsu)

        # Start the animation
        self.ani = FuncAnimation(self.fig, self.update_frame, frames=len(self.frames), interval=50, repeat=False)
        plt.show()

    def apply_otsu(self, event):
        """Apply Otsu method without freezing the application."""
        if self.otsu_applied or not self.animation_running:
            return

        self.otsu_applied = True

        def apply_otsu_async():
            nonlocal self
            current_frame = self.frames[self.current_frame_idx]
            histogram = Histogram(current_frame, 600, 800)
            binary_image = histogram.apply_threshold(current_frame)

            # Update the display with the binary image
            self.image.set_data(binary_image)
            self.ax.set_title("Otsu Threshold Applied")
            self.fig.canvas.draw_idle()

        otsu_thread = threading.Thread(target=apply_otsu_async)
        otsu_thread.start()

    def update_frame(self, frame_idx):
        """Update the frame in the animation."""
        if not self.animation_running or self.otsu_applied:
            return

        self.current_frame_idx = frame_idx
        self.image.set_data(self.frames[frame_idx])
        self.fig.canvas.draw_idle()

    def resume_simulation(self, event):
        """Resume the animation and generate new random data."""
        self.animation_running = True
        self.generator = DataGenerator(frame_shape=(600, 800), max_objects=10)
        self.frames = self.generator.generate_frames(num_frames=150)
        self.current_frame_idx = 0
        self.otsu_applied = False
        self.ani.frame_seq = self.ani.new_frame_seq()
        self.ani.event_source.start()

    def label_components(self, event):
        """Perform connected component labeling on the current frame."""
        if self.otsu_applied or not self.animation_running:
            return

        current_frame = self.frames[self.current_frame_idx]
        binary_image = current_frame > 0  # non-zero values represent objects

        ccl = ConnectedComponentLabeling(binary_image)
        labeled_image, num_labels = ccl.label_components(connectivity=8)

        # Update the display with the labeled image
        self.image.set_data(labeled_image)
        self.ax.set_title(f"Connected Components: {num_labels}")
        self.fig.canvas.draw_idle()       

    def count_blobs_in_otsu(self, event):
        """Count the blobs in the frame processed by the Otsu method."""
        if not self.otsu_applied or not self.animation_running:
            return

        current_frame = self.frames[self.current_frame_idx]
        histogram = Histogram(current_frame, 600, 800)
        binary_image = histogram.apply_threshold(current_frame)

        ccl = ConnectedComponentLabeling(binary_image)
        labeled_image, num_labels = ccl.label_components()
        num_blobs = ccl.count_blobs()

        self.ax.set_title(f"Labeled items amount: {num_blobs}") 
    