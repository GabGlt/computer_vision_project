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

@st.cache(allow_output_mutation=True)
def load_model(path="svm_final.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    bundle = load_model("svm_final.pkl")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

svm = bundle["model"]
scaler = bundle.get("scaler", None)
pca = bundle.get("pca", None)
class_names = bundle.get("class_names", svm.classes_)

def color_histogram(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist_hue = cv2.calcHist([image], [0], None, [180], [0, 180])
    hist_saturation = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_value = cv2.calcHist([image], [2], None, [256], [0, 256])

    cv2.normalize(hist_hue, hist_hue)
    cv2.normalize(hist_saturation, hist_saturation)
    cv2.normalize(hist_value, hist_value)

    color_feature_vector = np.concatenate([
        hist_hue.flatten(),
        hist_saturation.flatten(),
        hist_value.flatten()
    ]).astype(np.float32)

    return color_feature_vector

def ccm_feature(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=16):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H_channel, S_channel, V_channel = cv2.split(image)

    channels = {'H': H_channel, 'S': S_channel, 'V': V_channel}
    all_features = []

    for _, img_channel in channels.items():
        max_val = np.max(img_channel)
        if max_val > 0:
            quantized_img = (img_channel * (levels - 1) / max_val).astype(np.uint8)
        else:
            quantized_img = np.zeros_like(img_channel, dtype=np.uint8)

        glcm = graycomatrix(
            quantized_img,
            distances=distances,
            angles=angles,
            levels=levels,
            symmetric=True,
            normed=True
        )

        features_list = [
            graycoprops(glcm, prop).ravel()
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        ]

        channel_features = np.concatenate(features_list)
        all_features.append(channel_features)

    final_feature_vector = np.concatenate(all_features).astype(np.float32)
    return final_feature_vector

def sobel_feature(image, n_bins_mag=16, n_bins_ang=16):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    magnitude = np.sqrt(sobelx**2 + sobely**2)
    orientation = np.arctan2(sobely, sobelx)

    mag_flat = magnitude.ravel()
    ori_flat = orientation.ravel()

    if mag_flat.max() > 0:
        mag_flat = mag_flat / mag_flat.max()

    hist_mag, _ = np.histogram(
        mag_flat,
        bins=n_bins_mag,
        range=(0.0, 1.0),
        density=True
    )

    hist_ori, _ = np.histogram(
        ori_flat,
        bins=n_bins_ang,
        range=(-np.pi, np.pi),
        density=True
    )

    sobel_feat = np.concatenate([hist_mag, hist_ori]).astype(np.float32)
    return sobel_feat

def extract_features(img):
    f1 = color_histogram(img)
    f2 = ccm_feature(img)
    f3 = sobel_feature(img)
    return np.concatenate([f1,f2,f3])[None,:]

def preprocess(X):
    if scaler is not None:
        X = scaler.transform(X)
    if pca is not None:
        X = pca.transform(X)
    return X

def detect_feces_bbox(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )

    kernel = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(cnt)


menu = st.sidebar.radio("Menu", ["Home","Predict","Model Info"])

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Computer Vision Final Project**  \n"
    "Poultry Feces Classification using traditional Features Extraction & SVM"
)


if menu == "Home":
    st.markdown("## 🐔 Poultry Feces Classification System")

    st.write("""
    This web application presents a **computer vision–based classification system**
    designed to identify poultry feces conditions using **traditional features extraction**
    combined with **Principal Component Analysis (PCA)** and a **Support Vector Machine (SVM)** classifier.
    """)

    st.markdown("### 🔬 Classification Categories")
    st.markdown("""
    - **Coccidiosis**
    - **Salmonella**
    - **Healthy**
    """)

elif menu == "Predict":
    st.markdown("## 🔍 Image-based Prediction")
    with st.expander("📘 How to Use", expanded=False):
        st.markdown(
        """
        1. Upload an image containing poultry feces (**JPG / PNG** format).
        2. The system will automatically detect the feces region in the image.
        3. Visual features are extracted and processed using the trained **SVM** model.
        4. The predicted class and class probabilities will be displayed.
        """
    )


    uploaded = st.file_uploader(
        "Upload image (JPG / PNG)",
        ["jpg", "jpeg", "png"]
    )

    if uploaded:
        img_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        IMG_HEIGHT = 224
        IMG_WIDTH  = 224
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                 caption="Input Image", width=350)

        with st.spinner("Detecting feces region..."):
            bbox = detect_feces_bbox(img)

        if bbox is None:
            st.error("No feces detected in the image.")
            st.stop()

        x, y, w, h = bbox
        roi = img[y:y+h, x:x+w]

        X = extract_features(roi)
        X = preprocess(X)

        probs = svm.predict_proba(X)[0]
        idx = np.argmax(probs)
        pred_label = svm.classes_[idx]
        pred_class = class_names[int(pred_label)]

        boxed = img.copy()
        cv2.rectangle(boxed, (x, y), (x+w, y+h), (0, 255, 0), 3)

        st.image(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB),
                 caption="Detected ROI", width=350)

        st.markdown("### 🧪 Classification Result")
        st.success(f"**Predicted Condition:** {pred_class.upper()}")

        st.markdown("#### 📊 Class Probabilities")
        for i, p in enumerate(probs):
            st.write(f"- **{class_names[int(svm.classes_[i])]}** : {p:.4f}")

elif menu == "Model Info":
    st.markdown("## 📦 Model Information")

    st.markdown("### Class Labels")
    st.write(class_names)

    st.markdown("### Model Components")
    st.write(f"- **Classifier**: Support Vector Machine (SVM)")
    st.write(f"- **Scaler**: {type(scaler).__name__ if scaler else 'None'}")
    st.write(f"- **Dimensionality Reduction**: {type(pca).__name__ if pca else 'None'}")

    if pca:
        st.write(f"- **PCA Components**: {pca.n_components_}")
