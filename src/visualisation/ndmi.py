import numpy as np

def compute_ndmi(img):
    eps = 1e-6

    B8 = img[7].astype(np.float32) / 10000.0
    B11 = img[10].astype(np.float32) / 10000.0

    ndmi = (B8-B11)/(B8+B11)

    return ndmi