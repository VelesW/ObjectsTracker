import numpy as np
import cv2

class DataGenerator:
    def __init__(self, frame_shape=(600, 800), noise_mean=120, noise_std=40,
                 object_intensity=250, object_radius=6, max_objects=8, velocity_range=(-5, 5), shapes=("circle",)):
        """
        Initialize the radar simulation data generator with multiple shape support.

        :param frame_shape: Shape of the generated frames (height, width).
        :param noise_mean: Mean intensity for background noise.
        :param noise_std: Standard deviation for background noise.
        :param object_intensity: Intensity value for the objects (0-255).
        :param object_radius: Radius/size of the objects.
        :param max_objects: Maximum number of objects to be simulated.
        :param velocity_range: Range of velocities for the objects in both x and y directions.
        :param shapes: Tuple of shapes to include ("circle", "square", "triangle", "ellipse").
        """
        self.frame_shape = frame_shape
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.object_intensity = object_intensity
        self.object_radius = object_radius
        self.max_objects = max_objects
        self.velocity_range = velocity_range
        self.shapes = shapes + ("triangle", "square", "ellipse")
        self.objects = []
        self.velocities = []
        self.object_shapes = []

    def _initialize_objects(self, num_objects):
        """
        Randomly initialize the positions, velocities, and shapes of the objects.

        :param num_objects: Number of objects to initialize.
        """
        self.objects = np.array([
            np.random.randint(self.object_radius, self.frame_shape[dim] - self.object_radius, size=(num_objects,))
            for dim in range(2)
        ]).T
        self.velocities = np.random.randint(self.velocity_range[0], self.velocity_range[1] + 1, self.objects.shape)
        self.object_shapes = np.random.choice(self.shapes, size=num_objects)

    def _draw_shape(self, frame, position, shape):
        """
        Draw a shape on the frame.

        :param frame: The frame to draw on.
        :param position: Position (x, y) of the shape.
        :param shape: Type of shape ("circle", "square", "triangle", "ellipse").
        """
        if shape == "circle":
            cv2.circle(frame, tuple(position), self.object_radius, self.object_intensity, -1)
        elif shape == "square":
            top_left = (position[0] - self.object_radius, position[1] - self.object_radius)
            bottom_right = (position[0] + self.object_radius, position[1] + self.object_radius)
            cv2.rectangle(frame, top_left, bottom_right, self.object_intensity, -1)
        elif shape == "triangle":
            points = np.array([
                (position[0], position[1] - self.object_radius),
                (position[0] - self.object_radius, position[1] + self.object_radius),
                (position[0] + self.object_radius, position[1] + self.object_radius)
            ])
            cv2.fillPoly(frame, [points], self.object_intensity)
        elif shape == "ellipse":
            axes = (self.object_radius, self.object_radius // 2)
            cv2.ellipse(frame, tuple(position), axes, 0, 0, 360, self.object_intensity, -1)

    def _generate_frame(self):
        """
        Generate a single frame with moving objects and background noise.

        :return: A frame as a NumPy array.
        """
        # Create background noise
        frame = np.random.normal(self.noise_mean, self.noise_std, self.frame_shape).astype(np.uint8)

        # Update positions and draw objects
        for i in range(len(self.objects)):
            self.objects[i] += self.velocities[i]

            # Handle bouncing off the edges
            for dim in range(2):
                if self.objects[i][dim] - self.object_radius < 0 or self.objects[i][dim] + self.object_radius >= self.frame_shape[dim]:
                    self.velocities[i][dim] *= -1
                    self.objects[i][dim] = np.clip(self.objects[i][dim], self.object_radius, self.frame_shape[dim] - self.object_radius)

            # Draw the object on the frame
            self._draw_shape(frame, self.objects[i], self.object_shapes[i])

        return frame

    def generate_frames(self, num_frames, initial_objects=3):
        """
        Generate a sequence of frames with moving objects.

        :param num_frames: Total number of frames to generate.
        :param initial_objects: Number of objects to start the simulation with.
        :return: A list of frames as NumPy arrays.
        """
        self._initialize_objects(initial_objects)
        frames = []

        for t in range(num_frames):
            # Add new objects periodically
            if t > 0 and t % 5 == 0 and len(self.objects) < self.max_objects:
                new_object = np.random.randint(self.object_radius, self.frame_shape[1] - self.object_radius, size=2)
                new_shape = np.random.choice(self.shapes)
                self.objects = np.vstack([self.objects, new_object])
                new_velocity = np.random.randint(self.velocity_range[0], self.velocity_range[1] + 1, size=2)
                self.velocities = np.vstack([self.velocities, new_velocity])
                self.object_shapes = np.append(self.object_shapes, new_shape)

            frames.append(self._generate_frame())

        return frames
