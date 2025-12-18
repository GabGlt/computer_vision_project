import streamlit as st
import numpy as np
import pickle
import os
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
def load_model(path="best_svm_final.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    bundle = load_model("best_svm_final.pkl")
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

def auto_find_label(image_name):
    """
    cocci.0.jpg -> labels/cocci/cocci.0.txt
    """
    base = os.path.splitext(image_name)[0]  
    class_name = base.split(".")[0]           

    label_path = os.path.join("labels", class_name, base + ".txt")
    return label_path

def crop_roi_from_yolo(img, label_path):
    h, w, _ = img.shape

    if not os.path.exists(label_path):
        return None

    with open(label_path) as f:
        line = f.readline().strip()
        if line == "":
            return None
        _, xc, yc, bw, bh = map(float, line.split())

    xc *= w; yc *= h
    bw *= w; bh *= h

    x1 = int(xc - bw/2)
    y1 = int(yc - bh/2)
    x2 = int(xc + bw/2)
    y2 = int(yc + bh/2)

    x1 = max(0,x1); y1 = max(0,y1)
    x2 = min(w,x2); y2 = min(h,y2)

    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    return roi, (x1,y1,x2-x1,y2-y1)

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
            2. The system automatically loads the corresponding **YOLO label**.
            3. The feces region is cropped as ROI.
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

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                 caption="Input Image")
        
        label_path = auto_find_label(uploaded.name)

        if not os.path.exists(label_path):
            st.error(f"YOLO label not found:\n{label_path}")
            st.stop()

        result = crop_roi_from_yolo(img, label_path)
        if result is None:
            st.error("Failed to crop ROI")
            st.stop()

        roi, (x,y,w,h) = result
        roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))

        X = preprocess(extract_features(roi))
        probs = svm.predict_proba(X)[0]

        idx = np.argmax(probs)
        pred_class = class_names[idx]


        boxed = img.copy()
        cv2.rectangle(boxed, (x,y), (x+w,y+h), (0,255,0), 3)

        st.image(
            cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB),
            caption="Detected Feces Region",
            use_container_width=True
        )

        st.success(f"**Prediction: {pred_class.upper()}**")

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
