import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import convolve2d
import cv2

def load_kernels_from_mat(mat_file):
    data = loadmat(mat_file)
    for key in data:
        if not key.startswith("__") and isinstance(data[key], np.ndarray):
            arr = data[key]
            if arr.ndim == 3:  # Multiple 2D kernels: (num_kernels, k, k)
                return arr
            elif arr.ndim == 2:  # Single 2D kernel
                return np.expand_dims(arr, axis=0)
    raise ValueError(f"No valid 2D/3D kernel array found in {mat_file}")

def apply_kernel(image_gray, kernel):
    return convolve2d(image_gray, kernel, mode='same', boundary='symm')

def plot_kernels_and_filtered_images(image_gray, kernels, mat_filename, max_kernels=5, target_kernel_size=(15, 15)):
    n = min(max_kernels, len(kernels))
    
    plt.figure(figsize=(15, 4))
    for i in range(n):
        resized_kernel = resize_kernel(kernels[i], target_kernel_size)
        filtered = apply_kernel(image_gray, resized_kernel)

        plt.subplot(2, n, i+1)
        plt.imshow(resized_kernel, cmap='gray')
        plt.title(f"Kernel {i+1}")
        plt.axis('off')

        plt.subplot(2, n, i+1+n)
        plt.imshow(filtered, cmap='gray')
        plt.title(f"Filtered {i+1}")
        plt.axis('off')

    plt.suptitle(f"Kernels from: {os.path.basename(mat_filename)}", fontsize=14)
    plt.tight_layout()
    plt.show()

def resize_kernel(kernel, size=(3, 3)):
    return cv2.resize(kernel, size, interpolation=cv2.INTER_LINEAR)

def main(image_path, kernel_folder):
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_gray = cv2.resize(image_gray, (192, 64))  # Resize to 512x512
    if image_gray is None:
        raise FileNotFoundError(f"Image not found or invalid: {image_path}")
    image_gray = image_gray.astype(np.float32) / 255.0

    mat_files = [f for f in os.listdir(kernel_folder) if f.endswith('.mat')]
    
    for mat_file in mat_files:
        mat_path = os.path.join(kernel_folder, mat_file)
        try:
            kernels = load_kernels_from_mat(mat_path)
            plot_kernels_and_filtered_images(image_gray, kernels, mat_path)
        except Exception as e:
            print(f"[Skipped] {mat_file}: {e}")

if __name__ == "__main__":
    image_path = "data/Real-ESRGAN_result/15A50195_out.png"       # ← Thay bằng đường dẫn ảnh xám
    kernel_folder = "estimated-kn"  # ← Thay bằng thư mục chứa .mat
    main(image_path, kernel_folder)
