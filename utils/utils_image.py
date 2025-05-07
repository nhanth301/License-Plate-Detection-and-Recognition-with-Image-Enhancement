import numpy as np
from scipy.io import loadmat
import os
from scipy.signal import convolve2d
import cv2
def uint2single(img):

    return np.float32(img/255.)


def single2uint(img):

    return np.uint8((img.clip(0, 1)*255.).round())


def load_kernels_from_mat_folder(mat_folder, kernel_size=(15, 15)):
    mat_files = [f for f in os.listdir(mat_folder) if f.endswith('.mat')]   
    kernels = []
    for mat_file in mat_files:
        mat_path = os.path.join(mat_folder, mat_file)
        data = loadmat(mat_path)
        for key in data:
            if not key.startswith("__") and isinstance(data[key], np.ndarray):
                arr = data[key]
                # resize the kernel to 3x3
                arr = resize_kernel(arr, kernel_size)
                kernels.append(arr)
    return np.array(kernels)

def apply_kernel_rgb(img, kernel):
    return np.stack([
        cv2.filter2D(img[:, :, c], ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT)
        for c in range(3)
    ], axis=2)

def resize_kernel(kernel, size=(3, 3)):
    return cv2.resize(kernel, size, interpolation=cv2.INTER_LINEAR)

if __name__ == "__main__":
    mat_folder = "/home/nhan/Desktop/Plate/license-plate-super-resolution/estimated-kn"
    kernels = load_kernels_from_mat_folder(mat_folder)
    print(kernels.shape)