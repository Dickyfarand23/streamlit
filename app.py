import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ======================= Fungsi-Fungsi Pengolahan Citra =======================

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def gaussian_blur(image, ksize=5):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def apply_otsu_threshold(gray_image):
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def prewitt_edge(image):
    gray = to_grayscale(image)
    kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    x = cv2.filter2D(gray, -1, kernelx)
    y = cv2.filter2D(gray, -1, kernely)
    return cv2.addWeighted(x, 0.5, y, 0.5, 0)

def sobel_edge(gray_image):
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    return np.uint8(grad)

def histogram_equalization(gray_image):
    return cv2.equalizeHist(gray_image)

def quantize_compression(image, levels):
    div = 256 // levels
    quantized = image // div * div + div // 2
    return np.uint8(quantized)

# ======================= UI Streamlit =======================

st.set_page_config("Aplikasi Pengolahan Citra", layout="wide")
st.title("🖼️ Aplikasi Pengolahan Citra Interaktif")
st.markdown("Unggah gambar dan pilih metode pengolahan citra dari menu di sebelah kiri.")

st.sidebar.title("🔧 Pilih Metode")
method = st.sidebar.selectbox("Metode Pengolahan", [
    "Grayscale",
    "Gaussian Blur",
    "Otsu Thresholding",
    "Prewitt Edge Detection",
    "Sobel Edge Detection",
    "Histogram Equalization",
    "Quantizing Compression"
])

uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.subheader("Gambar Asli")
    st.image(image, use_column_width=True)

    result = None
    if method == "Grayscale":
        result = to_grayscale(image_bgr)
    elif method == "Gaussian Blur":
        result = gaussian_blur(image_bgr)
    elif method == "Otsu Thresholding":
        gray = to_grayscale(image_bgr)
        result = apply_otsu_threshold(gray)
    elif method == "Prewitt Edge Detection":
        result = prewitt_edge(image_bgr)
    elif method == "Sobel Edge Detection":
        gray = to_grayscale(image_bgr)
        result = sobel_edge(gray)
    elif method == "Histogram Equalization":
        gray = to_grayscale(image_bgr)
        result = histogram_equalization(gray)
    elif method == "Quantizing Compression":
        levels = st.sidebar.slider("Level Kuantisasi", 2, 64, 16)
        result = quantize_compression(image_bgr, levels)

    if result is not None:
        st.subheader(f"Hasil: {method}")
        st.image(result, use_column_width=True, clamp=True, channels="GRAY" if len(result.shape) == 2 else "BGR")

else:
    st.warning("⚠️ Harap unggah gambar terlebih dahulu.")
