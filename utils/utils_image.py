import numpy as np

def uint2single(img):

    return np.float32(img/255.)


def single2uint(img):

    return np.uint8((img.clip(0, 1)*255.).round())
