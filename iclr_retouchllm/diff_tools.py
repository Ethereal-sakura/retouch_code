import cv2
import numpy as np
from code_tools import load_img

def get_stat(img, load=True):
    if load:
        img = load_img(img) * 255.0
    img = np.array(img).astype(np.uint8)

    # rgb stat
    pixels = img.ravel()
    img_mean = round(np.mean(pixels), 2)
    img_median = round(np.median(pixels), 2)
    img_std = round(np.std(pixels), 2)
    img_percentail10 = round(np.percentile(pixels, 10), 2)
    img_percentail90 = round(np.percentile(pixels, 90), 2)
    rgb_mean = list(map(lambda x: round(x, 2), np.mean(img, axis=(0, 1))))

    # detail stat
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)  # Apply Laplacian filter
    laplacian_var = round(laplacian.var(), 2)

    # hsv stat
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s_mean = round(np.mean(hsv[..., 1]), 2)
    s_std = round(np.std(hsv[..., 1]), 2)
    s_min = round(np.min(hsv[..., 1]), 2)
    s_max = round(np.max(hsv[..., 1]), 2)

    # lab stat
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_mean = round(np.mean(lab[..., 0]), 2)
    b_mean = round(np.mean(lab[...,2]), 2)

    pixel_stat = {
        'pixel mean': img_mean,
        'pixel median': img_median,
        'pixel std': img_std,
        'pixel percentail 10%': img_percentail10,
        'pixel percentail 90%': img_percentail90,
        'rgb mean': rgb_mean,
        'laplacian variance': laplacian_var,
        'saturation mean': s_mean,
        'saturation std': s_std,
        'saturation min': s_min,
        'saturation max': s_max,
        'l-channel mean': l_mean,
        'b-channel mean': b_mean,
    }
    return pixel_stat
