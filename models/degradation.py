import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import random
import math
import matplotlib.pyplot as plt

class LPDegradationModel:
    def __init__(self):
        """Initialize the LP degradation model with configurable parameters"""
        # Default parameters
        self.gaussian_sigma_range = (0.5, 2.0)
        self.noise_level_range = (0.01, 0.05)
        self.motion_blur_kernel_size_range = (5, 15)
        self.brightness_weight_range = (0.3, 0.7)
    
    def apply_degradation(self, hr_image):
        """Apply the full degradation pipeline to convert HR image to LR
        
        Args:
            hr_image: High-resolution license plate image (RGB)
            
        Returns:
            lr_image: Degraded low-resolution image
        """
        # Make a copy of the input image
        img = hr_image.copy()
        
        # Apply domain-specific degradations (lighting and motion blur)
        apply_lighting = random.random() > 0.5
        apply_motion_blur = random.random() > 0.5
        
        if apply_lighting:
            img = self.apply_lighting_effect(img)
            
        if apply_motion_blur:
            img = self.apply_motion_blur(img)
        
        # Apply general degradations (Gaussian blur, noise)
        sigma = random.uniform(*self.gaussian_sigma_range)
        img = cv2.GaussianBlur(img, (0, 0), sigma)
        
        # Apply noise
        noise_level = random.uniform(*self.noise_level_range)
        img = self.apply_noise(img, noise_level)
        
        return img
    
    def apply_noise(self, img, noise_level):
        """Add random noise to the image - vectorized implementation
        
        Args:
            img: Input image
            noise_level: Level of noise to add (standard deviation)
            
        Returns:
            noisy_img: Image with added noise
        """
        # Convert to float32
        img_float = img.astype(np.float32) / 255.0
        
        # Generate noise
        noise = np.random.normal(0, noise_level, img_float.shape)
        
        # Add noise and clip values
        noisy_img = np.clip(img_float + noise, 0, 1) * 255
        
        return noisy_img.astype(np.uint8)
    
    def apply_motion_blur(self, img):
        """Apply motion blur using random PSF
        
        Args:
            img: Input image
            
        Returns:
            motion_blurred_img: Image with motion blur applied
        """
        # Generate random PSF kernel
        kernel_size = random.randint(*self.motion_blur_kernel_size_range)
        kernel = self._generate_motion_blur_kernel(kernel_size)
        
        # Apply convolution
        return cv2.filter2D(img, -1, kernel)
    
    def _generate_motion_blur_kernel(self, kernel_size):
        """Generate a motion blur kernel
        
        Args:
            kernel_size: Size of the kernel
            
        Returns:
            kernel: Motion blur kernel
        """
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        # Linear motion blur
        if random.random() > 0.5:
            angle = random.uniform(0, 360)
            length = random.uniform(kernel_size / 4, kernel_size / 2)
            
            # Create a line
            radian = np.deg2rad(angle)
            dx = math.cos(radian)
            dy = math.sin(radian)
            
            # Vectorized approach for line generation
            t = np.arange(int(length))
            x = (center + dx * t).astype(int)
            y = (center + dy * t).astype(int)
            
            # Filter valid points
            valid_points = (x >= 0) & (x < kernel_size) & (y >= 0) & (y < kernel_size)
            x, y = x[valid_points], y[valid_points]
            
            # Set values
            kernel[y, x] = 1
        else:
            # Non-linear motion blur (random trajectory)
            x, y = center, center
            
            # Random start angle
            angle = random.uniform(0, 360)
            prev_angle = angle
            
            # Generate points along a random trajectory
            points = [(x, y)]
            for _ in range(random.randint(5, 10)):
                # Change angle slightly from previous to ensure continuity
                angle_change = random.uniform(-30, 30)  # Max 30 degrees change
                angle = (prev_angle + angle_change) % 360
                prev_angle = angle
                
                # Calculate new position
                radian = np.deg2rad(angle)
                step_length = random.uniform(1, 2)
                x += math.cos(radian) * step_length
                y += math.sin(radian) * step_length
                
                if 0 <= int(y) < kernel_size and 0 <= int(x) < kernel_size:
                    points.append((x, y))
            
            # Draw the trajectory
            for x, y in points:
                kernel[int(y), int(x)] = 1
        
        # Normalize the kernel
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel = kernel / kernel_sum
            
        return kernel
    
    def _generate_ambient_light_mask(self, shape):
        """
        Generate a uniform light mask for ambient light.
        
        Args:
            shape (tuple): Image shape (height, width, channels).
        
        Returns:
            numpy.ndarray: Light mask with uniform intensity in [0.3, 1.0].
        """
        intensity = np.random.uniform(0.3, 0.7)  # Random intensity for practical effect
        return np.full(shape[:2], intensity, dtype=np.float32)

    def _generate_parallel_light_mask(self, shape):
        """
        Generate a light mask for parallel light with a directional gradient.
        
        Args:
            shape (tuple): Image shape (height, width, channels).
        
        Returns:
            numpy.ndarray: Light mask with Gaussian attenuation along a direction.
        """
        height, width = shape[:2]
        direction = np.random.choice(['horizontal', 'vertical'])
        
        if direction == 'horizontal':
            side = np.random.choice(['left', 'right'])
            if side == 'left':
                d = np.arange(width)  # Distance increases from left to right
            else:  # right
                d = width - 1 - np.arange(width)  # Distance decreases from left to right
            sigma = width / 1.5  # Gaussian spread based on image width
            mask_1d = np.exp(-d**2 / sigma**2)  # Gaussian attenuation
            mask = np.tile(mask_1d, (height, 1))  # Repeat across rows
        else:  # vertical
            side = np.random.choice(['top', 'bottom'])
            if side == 'top':
                d = np.arange(height)  # Distance increases from top to bottom
            else:  # bottom
                d = height - 1 - np.arange(height)  # Distance decreases from top to bottom
            sigma = height / 1.5  # Gaussian spread based on image height
            mask_1d = np.exp(-d**2 / sigma**2)  # Gaussian attenuation
            mask = np.tile(mask_1d[:, np.newaxis], (1, width))  # Repeat across columns
        
        return mask.astype(np.float32)

    def _generate_spotlight_mask(self, shape):
        """
        Generate a light mask for spotlight with radial attenuation.
        
        Args:
            shape (tuple): Image shape (height, width, channels).
        
        Returns:
            numpy.ndarray: Light mask with Gaussian attenuation from a point.
        """
        height, width = shape[:2]
        # Random light source position
        x0 = np.random.randint(0, width)
        y0 = np.random.randint(0, height)
        # Create coordinate grid
        i, j = np.mgrid[0:height, 0:width]
        # Compute radial distance from light source
        d = np.sqrt((i - y0)**2 + (j - x0)**2)
        sigma = max(width, height) / 1.5  # Gaussian spread based on max dimension
        mask = np.exp(-d**2 / sigma**2)  # Gaussian attenuation
        return mask.astype(np.float32)

    def apply_lighting_effect(self, image):
        """
        Apply a random lighting effect (ambient, parallel, or spotlight) to an image.
        
        Args:
            image (numpy.ndarray): Input RGB image with shape (height, width, 3) and dtype uint8.
        
        Returns:
            numpy.ndarray: Output RGB image with lighting effect applied.
        """
        # Randomly select a lighting effect
        effect = np.random.choice(['ambient', 'parallel', 'spotlight'])
        
        # Generate the corresponding light mask
        if effect == 'ambient':
            light_mask = self._generate_ambient_light_mask(image.shape)
        elif effect == 'parallel':
            light_mask = self._generate_parallel_light_mask(image.shape)
        else:  # spotlight
            light_mask = self._generate_spotlight_mask(image.shape)
        
        # Convert image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Apply light mask to the V channel
        v_channel = hsv_image[:, :, 2].astype(np.float32)  # Convert to float for multiplication
        v_channel = (v_channel * light_mask).clip(0, 255).astype(np.uint8)  # Apply mask and clip
        
        # Update the V channel in the HSV image
        hsv_image[:, :, 2] = v_channel
        
        # Convert back to RGB
        result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        
        return result_image


def batch_process_degradations(hr_image, num_variations=10):
    """Process multiple degradations in parallel
    
    Args:
        hr_image: High-resolution input image
        num_variations: Number of degraded versions to create
        
    Returns:
        degraded_images: List of degraded images
    """
    degradation_model = LPDegradationModel()
    degraded_images = []
    
    for _ in range(num_variations):
        lr_image = degradation_model.apply_degradation(hr_image)
        degraded_images.append(lr_image)
        
    return degraded_images


if __name__ == "__main__":
    hr_image = cv2.imread("data/test/sub_2nd_20m_M09_D28_C166_14A74506_0.62_1693306799412.png")

    if hr_image is None:
        print("Error: Could not load image")
    else:
        hr_image_rgb = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        degraded_images = batch_process_degradations(hr_image_rgb, 10)

        plt.figure(figsize=(15, 8))

        plt.subplot(3, 4, 1)
        plt.imshow(hr_image_rgb)
        plt.title("Original HR Image")
        plt.axis("off")

        for i in range(10):
            plt.subplot(3, 4, i + 2)
            plt.imshow(degraded_images[i])
            print(degraded_images[i].shape)
            plt.title(f"Degraded {i+1}")
            plt.axis("off")
        plt.tight_layout()
        save_path = "results/degradation/sample01.png"  # Change to your desired path
        # plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
