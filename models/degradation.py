import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import cv2
import random
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.utils_image import uint2single, single2uint

class LPDegradationModel:
    """
    A class to simulate low-resolution images by applying degradations to high-resolution images.
    Degradations include lighting effects, motion blur, Gaussian blur, scaling, and noise.
    """
    def __init__(self):
        """
        Initialize the degradation model with hyperparameter ranges.
        All ranges are set to ensure reasonable degradation effects while output remains in [0,1] via clipping.
        """
        self.gaussian_sigma_range = (4.0, 6.0)  # Sigma range for Gaussian blur intensity
        self.noise_level_range = (0.01, 0.05)   # Noise standard deviation range
        self.motion_blur_kernel_size_range = (17, 23)  # Kernel size range for motion blur
        self.brightness_weight_range = (0.3, 0.7)  # Intensity range for lighting effects
    
    def apply_degradation(self, hr_image):
        """
        Apply a fixed sequence of degradations to the input high-resolution image.

        Args:
            hr_image (numpy.ndarray): High-resolution image with values in [0,1].

        Returns:
            numpy.ndarray: Degraded image with values clipped to [0,1].
        """
        img = hr_image.copy()
        
        # Apply lighting effect with 70% probability
        if random.random() > 0.7:
            img = self.apply_lighting_effect(img)
        
        # Apply motion blur with 70% probability
        if random.random() > 0.7:
            img = self.apply_motion_blur(img)
        
        # Apply Gaussian blur with random sigma
        sigma = random.uniform(*self.gaussian_sigma_range)
        img = np.clip(cv2.GaussianBlur(img, (0, 0), sigma), 0, 1)
        
        # Scale down the image
        img = self.scale_down(img, 0.25)
        
        # Add noise with random level
        noise_level = random.uniform(*self.noise_level_range)
        img = self.apply_noise(img, noise_level)
        
        return np.clip(img, 0, 1)  # Ensure final output is in [0,1]

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

def batch_process_degradations(hr_image, num_variations=10):
    """
    Generate multiple degraded versions of the input image.

    Args:
        hr_image (numpy.ndarray): High-resolution image in [0,1].
        num_variations (int): Number of degraded images to generate.

    Returns:
        list: List of degraded images.
    """
    degradation_model = LPDegradationModel()
    degraded_images = []
    for _ in range(num_variations):
        lr_image = degradation_model.apply_degradation(hr_image)
        degraded_images.append(lr_image)
    return degraded_images

if __name__ == "__main__":
    output_dir = "results/degradation/images"
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Load the high-resolution image
    hr_image = cv2.imread("/home/anhnh/Downloads/ccpd_cropped/val/浙E3Z933.jpg")
    if hr_image is None:
        print("Error: Could not load image")
    else:
        # Convert to RGB and [0,1] float range
        hr_image_rgb = uint2single(cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB))
        
        # Generate 24 degraded versions
        degraded_images = batch_process_degradations(hr_image_rgb, 24)

        # Save the original HR image
        hr_image_uint8 = single2uint(hr_image_rgb)  # Convert [0,1] float to [0,255] uint8
        cv2.imwrite(os.path.join(output_dir, "original_hr_image.png"), cv2.cvtColor(hr_image_uint8, cv2.COLOR_RGB2BGR))

        # Save each degraded image
        for i, degraded_img in enumerate(degraded_images):
            degraded_img_uint8 = single2uint(degraded_img)  # Convert [0,1] float to [0,255] uint8
            filename = os.path.join(output_dir, f"degraded_{i+1:02d}.png")
            cv2.imwrite(filename, cv2.cvtColor(degraded_img_uint8, cv2.COLOR_RGB2BGR))
            print(f"Saved {filename} with shape {degraded_img.shape}")

        # Optional: Create and save the visualization plot
        plt.figure(figsize=(20, 20))
        plt.subplot(5, 5, 1)
        plt.imshow(hr_image_rgb)
        plt.title("Original HR Image")
        plt.axis("off")
        for i in range(24):
            plt.subplot(5, 5, i + 2)
            plt.imshow(degraded_images[i])
            plt.title(f"Degraded {i+1}")
            plt.axis("off")
        plt.tight_layout()
        plt.savefig("results/degradation/degradation_plot.png", dpi=300, bbox_inches="tight")
        plt.close()