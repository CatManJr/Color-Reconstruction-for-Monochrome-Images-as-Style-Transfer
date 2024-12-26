import cv2
import numpy as np
from PIL import Image

def bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    img = np.array(img)
    filtered_img = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    return Image.fromarray(filtered_img)