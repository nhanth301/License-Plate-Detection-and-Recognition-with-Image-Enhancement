import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import cv2
import random
import math
import matplotlib.pyplot as plt
from utils.utils_image import uint2single, single2uint, load_kernels_from_mat_folder, apply_kernel_rgb

class LPDegradationModel:
    """
    A class to simulate low-resolution images by applying degradations to high-resolution images.
    Degradations include lighting effects, motion blur, Gaussian blur, scaling, and noise.
    """
    def __init__(self, gaussian_sigma_range=(4.0, 6.0), noise_level_range=(0.01, 0.05), motion_blur_kernel_size_range=(17, 23), brightness_weight_range=(0.3, 0.7), lr_size=(128,64), scale=0.35):
        """
        Initialize the degradation model with hyperparameter ranges.
        All ranges are set to ensure reasonable degradation effects while output remains in [0,1] via clipping.
        """
        self.gaussian_sigma_range = gaussian_sigma_range  # Sigma range for Gaussian blur intensity
        self.noise_level_range = noise_level_range   # Noise standard deviation range
        self.motion_blur_kernel_size_range = motion_blur_kernel_size_range  # Kernel size range for motion blur
        self.brightness_weight_range = brightness_weight_range  # Intensity range for lighting effects
        self.scale = scale  
        self.lr_size = lr_size
        self.kernels = load_kernels_from_mat_folder("/home/nhan/Desktop/Plate/license-plate-super-resolution/estimated-kn")
    def apply_degradation(self, hr_image):
        """
        Apply a fixed sequence of degradations to the input high-resolution image.

        Args:
            hr_image (numpy.ndarray): High-resolution image with values in [0,1].

        Returns:
            numpy.ndarray: Degraded image with values clipped to [0,1].
        """
        img = hr_image.copy()
        # random kernel
        kernel_index = random.randint(0, len(self.kernels)-1)
        img = apply_kernel_rgb(img, self.kernels[kernel_index])

        
        if random.random() > 0.7:
            img = self.apply_motion_blur(img)
        
        # Apply Gaussian blur with random sigma
        sigma = random.uniform(*self.gaussian_sigma_range)
        img = np.clip(cv2.GaussianBlur(img, (0, 0), sigma), 0, 1)
        
        # Scale down the image
        img = self.scale_down(img, self.scale)
        
        # Add noise with random level
        noise_level = random.uniform(*self.noise_level_range)
        img = self.apply_noise(img, noise_level)
        
        return cv2.resize(np.clip(img, 0, 1),self.lr_size)  # Ensure final output is in [0,1]

    def scale_down(self, img, scale_factor, interpolation="bicubic"):
        """
        Reduce the image size by a given scale factor.

        Args:
            img (numpy.ndarray): Input image in [0,1].
            scale_factor (float): Factor between 0 and 1 to reduce size.
            interpolation (str): Interpolation method ('bicubic', 'bilinear', 'nearest').

        Returns:
            numpy.ndarray: Scaled-down image clipped to [0,1].
        """
        if scale_factor >= 1.0 or scale_factor <= 0:
            raise ValueError("Scale factor must be between 0 and 1.")
        interpolation_methods = {
            "bicubic": cv2.INTER_CUBIC,
            "bilinear": cv2.INTER_LINEAR,
            "nearest": cv2.INTER_NEAREST
        }
        new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
        downscaled_img = cv2.resize(img, new_size, interpolation=interpolation_methods[interpolation])
        return np.clip(downscaled_img, 0, 1)  # Clip to maintain [0,1] range

    def apply_noise(self, img, noise_level):
        """
        Add Gaussian noise to the image.

        Args:
            img (numpy.ndarray): Input image in [0,1].
            noise_level (float): Standard deviation of the Gaussian noise.

        Returns:
            numpy.ndarray: Noisy image clipped to [0,1].
        """
        noise = np.random.normal(0, noise_level, img.shape)
        noisy_img = img + noise
        return np.clip(noisy_img, 0, 1)  # Clip to ensure values stay in [0,1]

    def apply_motion_blur(self, img):
        """
        Apply motion blur to simulate camera movement.

        Args:
            img (numpy.ndarray): Input image in [0,1].

        Returns:
            numpy.ndarray: Blurred image clipped to [0,1].
        """
        kernel_size = random.randint(*self.motion_blur_kernel_size_range)
        kernel = self._generate_motion_blur_kernel(kernel_size)
        blurred_img = cv2.filter2D(img, -1, kernel)
        return np.clip(blurred_img, 0, 1)  # Clip to maintain [0,1] range

    def _generate_motion_blur_kernel(self, kernel_size):
        """
        Create a motion blur kernel based on random direction or path.

        Args:
            kernel_size (int): Size of the kernel.

        Returns:
            numpy.ndarray: Normalized kernel for motion blur.
        """
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        if random.random() > 0.5:
            # Linear motion blur
            angle = random.uniform(0, 360)
            length = random.uniform(kernel_size / 4, kernel_size / 2)
            radian = np.deg2rad(angle)
            dx, dy = math.cos(radian), math.sin(radian)
            t = np.arange(int(length))
            x = (center + dx * t).astype(int)
            y = (center + dy * t).astype(int)
            valid_points = (x >= 0) & (x < kernel_size) & (y >= 0) & (y < kernel_size)
            x, y = x[valid_points], y[valid_points]
            kernel[y, x] = 1
        else:
            # Random walk motion blur
            x, y = center, center
            angle = random.uniform(0, 360)
            prev_angle = angle
            points = [(x, y)]
            for _ in range(random.randint(5, 10)):
                angle_change = random.uniform(-30, 30)
                angle = (prev_angle + angle_change) % 360
                prev_angle = angle
                radian = np.deg2rad(angle)
                step_length = random.uniform(1, 2)
                x += math.cos(radian) * step_length
                y += math.sin(radian) * step_length
                if 0 <= int(y) < kernel_size and 0 <= int(x) < kernel_size:
                    points.append((x, y))
            for x, y in points:
                kernel[int(y), int(x)] = 1
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel = kernel / kernel_sum  # Normalize kernel
        return kernel

    def _generate_ambient_light_mask(self, shape):
        """
        Generate a uniform ambient light mask.

        Args:
            shape (tuple): Shape of the image (height, width, channels).

        Returns:
            numpy.ndarray: Ambient light mask.
        """
        intensity = np.random.uniform(*self.brightness_weight_range)
        return np.full(shape[:2], intensity, dtype=np.float32)

    def _generate_parallel_light_mask(self, shape):
        """
        Generate a parallel light mask with gradient effect.

        Args:
            shape (tuple): Shape of the image.

        Returns:
            numpy.ndarray: Parallel light mask.
        """
        height, width = shape[:2]
        direction = np.random.choice(['horizontal', 'vertical'])
        if direction == 'horizontal':
            side = np.random.choice(['left', 'right'])
            d = np.arange(width) if side == 'left' else width - 1 - np.arange(width)
            sigma = width / 1.5
            mask_1d = np.exp(-d**2 / sigma**2)
            mask = np.tile(mask_1d, (height, 1))
        else:
            side = np.random.choice(['top', 'bottom'])
            d = np.arange(height) if side == 'top' else height - 1 - np.arange(height)
            sigma = height / 1.5
            mask_1d = np.exp(-d**2 / sigma**2)
            mask = np.tile(mask_1d[:, np.newaxis], (1, width))
        return mask.astype(np.float32)

    def _generate_spotlight_mask(self, shape):
        """
        Generate a spotlight mask centered at a random point.

        Args:
            shape (tuple): Shape of the image.

        Returns:
            numpy.ndarray: Spotlight mask.
        """
        height, width = shape[:2]
        x0, y0 = np.random.randint(0, width), np.random.randint(0, height)
        i, j = np.mgrid[0:height, 0:width]
        d = np.sqrt((i - y0)**2 + (j - x0)**2)
        sigma = max(width, height) / 1.5
        mask = np.exp(-d**2 / sigma**2)
        return mask.astype(np.float32)

    def apply_lighting_effect(self, image):
        """
        Apply a random lighting effect (ambient, parallel, or spotlight) to the image.

        Args:
            image (numpy.ndarray): Input image in [0,1].

        Returns:
            numpy.ndarray: Image with lighting effect, clipped to [0,1].
        """
        effect = np.random.choice(['ambient', 'parallel', 'spotlight'])
        if effect == 'ambient':
            light_mask = self._generate_ambient_light_mask(image.shape)
        elif effect == 'parallel':
            light_mask = self._generate_parallel_light_mask(image.shape)
        else:
            light_mask = self._generate_spotlight_mask(image.shape)
        image = image.astype(np.float32)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        v_channel = hsv_image[:, :, 2]
        v_channel = np.clip(v_channel * light_mask, 0, 1)  # Clip value channel
        hsv_image[:, :, 2] = v_channel
        result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        return np.clip(result_image, 0, 1)  # Clip final RGB output

if __name__ == "__main__":
    degradation = LPDegradationModel()
    img = cv2.imread("/home/nhan/Desktop/New Folder/HR/_2_jpg.rf.1862bf8d42707d677b75a37396adb1d1_out.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = uint2single(img)
    degraded_img = degradation.apply_degradation(img)
    degraded_img = single2uint(degraded_img)
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(degraded_img)
    plt.title("Degraded Image")
    plt.axis('off')
    plt.show()
    cv2.imwrite("degraded_image.png", degraded_img)