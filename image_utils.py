import cv2
import numpy as np

# Fungsi untuk mengubah citra berwarna menjadi grayscale
# def to_grayscale(image):
#     return ...


# Fungsi untuk menerapkan Gaussian blur
# def gaussian_blur(image, ksize=(3,3)):
#     return ...


# Fungsi untuk menerapkan thresholding Otsu pada citra grayscale
# def apply_otsu_threshold(gray):
#     _, thresh = cv2.threshold(...)
#     return ...


# Fungsi untuk mendeteksi tepi menggunakan operator Prewitt
# def prewitt_edge(image):
#     kernelx = np.array([...])
#     kernely = np.array([...])
#     x = cv2.filter2D(...)
#     y = cv2.filter2D(...)
#     return ...


# Fungsi untuk mendeteksi tepi menggunakan operator Sobel
# def sobel_edge(image):
#     grad_x = cv2.Sobel(...)
#     grad_y = cv2.Sobel(...)
#     abs_grad_x = cv2.convertScaleAbs(...)
#     abs_grad_y = cv2.convertScaleAbs(...)
#     return cv2.addWeighted(....)


# Fungsi untuk melakukan histogram equalization
# def histogram_equalization(image):
#     return ...


# Fungsi untuk mengompresi citra grayscale menggunakan kuantisasi
# def quantize_compression(image, levels=16):
#     image = cv2.cvtColor(...) if len(image.shape) == 3 else image
#     step = ...
#     compressed = ...
#     return compressed.astype(...)

