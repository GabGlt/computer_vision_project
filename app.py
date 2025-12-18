import streamlit as st
import numpy as np
import pickle
import cv2
from skimage.feature import graycomatrix, graycoprops

st.set_page_config(
    page_title="Poultry Feces Classifier",
    page_icon="🐔",
    layout="wide"
)

IMG_SIZE = 256
LABEL_ROOT = "labels" 

@st.cache_resource
def load_model(path="best_svm.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    bundle = load_model("best_svm.pkl")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

svm         = bundle["model"]
scaler      = bundle.get("scaler", None)
pca         = bundle.get("pca", None)
class_names = bundle["class_names"]

def color_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h = cv2.calcHist([hsv],[0],None,[180],[0,180])
    s = cv2.calcHist([hsv],[1],None,[256],[0,256])
    v = cv2.calcHist([hsv],[2],None,[256],[0,256])

    cv2.normalize(h, h)
    cv2.normalize(s, s)
    cv2.normalize(v, v)

    return np.concatenate([h.flatten(), s.flatten(), v.flatten()]).astype(np.float32)

def ccm_feature(image, levels=16):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    feats = []

    for channel in cv2.split(hsv):
        max_val = channel.max()
        if max_val > 0:
            q = (channel * (levels-1) / max_val).astype(np.uint8)
        else:
            q = np.zeros_like(channel, dtype=np.uint8)

        glcm = graycomatrix(
            q,
            distances=[1],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=levels,
            symmetric=True,
            normed=True
        )

        props = ["contrast","dissimilarity","homogeneity","energy","correlation"]
        feats.append(np.concatenate([graycoprops(glcm,p).ravel() for p in props]))

    return np.concatenate(feats).astype(np.float32)

def sobel_feature(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    mag = np.sqrt(sx**2 + sy**2)
    ang = np.arctan2(sy, sx)

    if mag.max() > 0:
        mag /= mag.max()

    h_mag, _ = np.histogram(mag.ravel(), bins=16, range=(0,1), density=True)
    h_ang, _ = np.histogram(ang.ravel(), bins=16, range=(-np.pi,np.pi), density=True)

    return np.concatenate([h_mag, h_ang]).astype(np.float32)

def extract_features(img):
    f = np.concatenate([
        color_histogram(img),
        ccm_feature(img),
        sobel_feature(img)
    ])
    return f[None, :]

def preprocess(X):
    if scaler is not None:
        X = scaler.transform(X)
    if pca is not None:
        X = pca.transform(X)
    return X

def detect_feces_bbox(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
   
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )

    kernel = np.ones((5,5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w*h
        if area < 200:
            continue
        ratio = w / h
        if ratio < 0.2 or ratio > 5:
            continue
        patch = gray[y:y+h, x:x+w]
        if patch.size == 0:
            continue
        variance = np.var(patch)
        if variance < 20:
            continue
        candidates.append((c, area))

    if len(candidates) == 0:
        return None

    cnt = max(candidates, key=lambda x: x[1])[0]
    x, y, w, h = cv2.boundingRect(cnt)

    pad_w = int(w * 0.1)
    pad_h = int(h * 0.1)
    x = max(0, x - pad_w)
    y = max(0, y - pad_h)
    w = min(image.shape[1] - x, w + 2*pad_w)
    h = min(image.shape[0] - y, h + 2*pad_h)

    return x, y, w, h


menu = st.sidebar.radio("Menu", ["Home","Predict","Model Info"])

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Computer Vision Final Project**  \n"
    "Poultry Feces Classification using Traditional Features Extraction & SVM"
)


if menu == "Home":
    st.markdown(
        """
        <div class="header">
            <h1>🐔 Poultry Feces Classification</h1>
            <p>Detection and classification of poultry diseases from feces images</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🔬 Methodology")
    st.markdown("""
    This application applies a **traditional computer vision approach** consisting of:
    - **Color Histogram (HSV)** for color distribution
    - **GLCM (CCM)** for texture analysis
    - **Sobel Edge Histogram** for shape information  
    - **Principal Component Analysis (PCA)** for dimensionality reduction  
    - **Support Vector Machine (SVM)** for classification
    """)

    st.markdown("### 🦠 Target Classes")
    for cls in class_names:
        st.write(f"- {cls.capitalize()}")

elif menu == "Predict":
    st.markdown(
        """
        <div class="header">
            <h1>🔍 Image Prediction</h1>
            <p>Automatic feces detection and disease classification</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("📘 How to Use", expanded=False):
        st.markdown(
            """
            1. Upload a poultry feces image (**JPG / PNG**).
            2. The system automatically detects the feces region (ROI) using image processing techniques.
            3. Features are extracted from the detected ROI (color, texture, and shape).
            4. The trained **SVM model** predicts the disease class.
            """
        )


    uploaded = st.file_uploader(
        "Upload poultry feces image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        img_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("Failed to read image")
            st.stop()

        display_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        bbox = detect_feces_bbox(display_img)
        if bbox is None:
            st.error("Tidak ada objek terdeteksi")
            st.stop()

        x, y, w, h = bbox
        roi = display_img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))

        # Feature extraction & prediction
        X = preprocess(extract_features(roi))
        probs = svm.predict_proba(X)[0]
        idx = np.argmax(probs)
        pred_class = class_names[idx]

        # Draw bbox
        boxed = display_img.copy()
        cv2.rectangle(boxed, (x, y), (x+w, y+h), (0,255,0), 2)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), caption="Input Image", width=250)
        with col2:
            st.image(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB), caption=f"Detected ROI – Prediction: {pred_class.upper()}", width=250)

        # Probabilities
        st.markdown("### 📊 Probabilities")
        for cls, p in zip(class_names, probs):
            st.write(f"- **{cls}** : {p*100:.2f}%")

elif menu == "Model Info":
    st.markdown(
        """
        <div class="header">
            <h1>📦 Model Information</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🧠 Model Components")
    st.write(f"- **Classes**: {class_names}")
    st.write(f"- **Classifier**: {type(svm).__name__}")
    st.write(f"- **Scaler**: {type(scaler).__name__ if scaler else 'None'}")
    st.write(f"- **PCA**: {type(pca).__name__ if pca else 'None'}")
