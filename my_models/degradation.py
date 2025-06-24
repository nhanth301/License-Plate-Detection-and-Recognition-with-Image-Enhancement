import sys
import os
import math
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from my_utils.utils import (
    apply_kernel_rgb,
    load_kernels_from_mat_folder,
    single2uint,
    uint2single,
)


class LPDegradationModel:
    def __init__(
        self,
        gaussian_sigma_range=(1.5, 3.0),
        noise_level_range=(0.01, 0.02),
        motion_blur_kernel_size_range=(7, 13),
        brightness_weight_range=(0.3, 0.5),
        lr_size=(192, 32),
        scale=0.35,
    ):
        self.gaussian_sigma_range = gaussian_sigma_range
        self.noise_level_range = noise_level_range
        self.motion_blur_kernel_size_range = motion_blur_kernel_size_range
        self.brightness_weight_range = brightness_weight_range
        self.scale = scale
        self.lr_size = lr_size
        self.kernels = load_kernels_from_mat_folder(
            "/home/nhan/Desktop/Plate/license-plate-super-resolution/estimated-kn"
        )

    def apply_degradation(self, hr_image):
        img = hr_image.copy()
        # img = cv2.resize(img,(self.lr_size[0]*2, self.lr_size[1]*2), interpolation=cv2.INTER_CUBIC)
        # random kernel
        # kernel_index = random.randint(0, len(self.kernels)-1)
        # img = apply_kernel_rgb(img, self.kernels[kernel_index])
        # img = cv2.resize(np.clip(img, 0, 1),self.lr_size)

        if random.random() > 0.3:
            img = self.apply_motion_blur(img)

        if random.random() > 0.7:
            img = self.apply_lighting_effect(img)

        sigma = random.uniform(*self.gaussian_sigma_range)
        img = np.clip(cv2.GaussianBlur(img, (0, 0), sigma), 0, 1)

        img = self.scale_down(img, self.scale)

        noise_level = random.uniform(*self.noise_level_range)
        img = self.apply_noise(img, noise_level)

        return cv2.resize(np.clip(img, 0, 1), self.lr_size)

    def scale_down(self, img, scale_factor, interpolation="bicubic"):
        if scale_factor >= 1.0 or scale_factor <= 0:
            raise ValueError("Scale factor must be between 0 and 1.")
        interpolation_methods = {
            "bicubic": cv2.INTER_CUBIC,
            "bilinear": cv2.INTER_LINEAR,
            "nearest": cv2.INTER_NEAREST,
        }
        new_size = (
            int(img.shape[1] * scale_factor),
            int(img.shape[0] * scale_factor),
        )
        downscaled_img = cv2.resize(
            img, new_size, interpolation=interpolation_methods[interpolation]
        )
        return np.clip(downscaled_img, 0, 1)

    def apply_noise(self, img, noise_level):
        noise = np.random.normal(0, noise_level, img.shape)
        noisy_img = img + noise
        return np.clip(noisy_img, 0, 1)

    def apply_motion_blur(self, img):
        kernel_size = random.randint(*self.motion_blur_kernel_size_range)
        kernel = self._generate_motion_blur_kernel(kernel_size)
        blurred_img = cv2.filter2D(img, -1, kernel)
        return np.clip(blurred_img, 0, 1)

    def _generate_motion_blur_kernel(self, kernel_size):
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        if random.random() > 0.5:
            angle = random.uniform(0, 360)
            length = random.uniform(kernel_size / 4, kernel_size / 2)
            radian = np.deg2rad(angle)
            dx, dy = math.cos(radian), math.sin(radian)
            t = np.arange(int(length))
            x = (center + dx * t).astype(int)
            y = (center + dy * t).astype(int)
            valid_points = (
                (x >= 0) & (x < kernel_size) & (y >= 0) & (y < kernel_size)
            )
            x, y = x[valid_points], y[valid_points]
            kernel[y, x] = 1
        else:
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
            kernel = kernel / kernel_sum
        return kernel

    def _generate_ambient_light_mask(self, shape):
        intensity = np.random.uniform(*self.brightness_weight_range)
        return np.full(shape[:2], intensity, dtype=np.float32)

    def _generate_parallel_light_mask(self, shape):
        height, width = shape[:2]
        direction = np.random.choice(["horizontal", "vertical"])
        if direction == "horizontal":
            side = np.random.choice(["left", "right"])
            d = np.arange(width) if side == "left" else width - 1 - np.arange(width)
            sigma = width / 1.5
            mask_1d = np.exp(-(d**2) / sigma**2)
            mask = np.tile(mask_1d, (height, 1))
        else:
            side = np.random.choice(["top", "bottom"])
            d = (
                np.arange(height)
                if side == "top"
                else height - 1 - np.arange(height)
            )
            sigma = height / 1.5
            mask_1d = np.exp(-(d**2) / sigma**2)
            mask = np.tile(mask_1d[:, np.newaxis], (1, width))
        return mask.astype(np.float32)

    def _generate_spotlight_mask(self, shape):
        height, width = shape[:2]
        x0, y0 = np.random.randint(0, width), np.random.randint(0, height)
        i, j = np.mgrid[0:height, 0:width]
        d = np.sqrt((i - y0) ** 2 + (j - x0) ** 2)
        sigma = max(width, height) / 1.5
        mask = np.exp(-(d**2) / sigma**2)
        return mask.astype(np.float32)

    def apply_lighting_effect(self, image):
        effect = np.random.choice(["ambient", "parallel", "spotlight"])
        if effect == "ambient":
            light_mask = self._generate_ambient_light_mask(image.shape)
        elif effect == "parallel":
            light_mask = self._generate_parallel_light_mask(image.shape)
        else:
            light_mask = self._generate_spotlight_mask(image.shape)
        image = image.astype(np.float32)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        v_channel = hsv_image[:, :, 2]
        v_channel = np.clip(v_channel * light_mask, 0, 1)
        hsv_image[:, :, 2] = v_channel
        result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        return np.clip(result_image, 0, 1)


if __name__ == "__main__":
    degradation = LPDegradationModel()
    img = cv2.imread(
        "/home/nhan/Downloads/archive/cropped_images/train/carlong_0045_plate_1.png"
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = uint2single(img)
    degraded_img = degradation.apply_degradation(img)
    degraded_img = single2uint(degraded_img)
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(degraded_img)
    plt.title("Degraded Image")
    plt.axis("off")
    plt.show()
    cv2.imwrite("degraded_image.png", degraded_img)