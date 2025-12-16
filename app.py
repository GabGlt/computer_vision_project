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
def load_model(path="svm_model_nopca.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    bundle = load_model("svm_model_nopca.pkl")
except Exception as e:
    st.error(f"Fail to load model: {e}")
    st.stop()

svm = bundle["model"]
scaler = bundle["scaler"]
class_names = svm.classes_

label_mapping = {
    0: "cocci",
    1: "salmo",
    2: "healthy"
}


def color_histogram(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([image], [0], None, [180], [0,180])
    s = cv2.calcHist([image], [1], None, [256], [0,256])
    v = cv2.calcHist([image], [2], None, [256], [0,256])
    cv2.normalize(h, h)
    cv2.normalize(s, s)
    cv2.normalize(v, v)
    return np.concatenate((h.flatten(), s.flatten(), v.flatten())).astype(np.float32)

def ccm_feature(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(image)
    channels = [H, S, V]
    features = []   
    for ch in channels:
        lvl = 16
        if ch.max() > 0:
            q = (ch * (lvl-1) / ch.max()).astype(np.uint8)
        else:
            q = np.zeros_like(ch, dtype=np.uint8)

        glcm = graycomatrix(
            q, distances=[1],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=lvl, symmetric=True,
            normed=True
        )
        props = ['contrast','dissimilarity','homogeneity','energy','correlation']
        feat = np.concatenate([graycoprops(glcm, p).ravel() for p in props])
        features.append(feat)
    return np.concatenate(features).astype(np.float32)

def sobel_feature(image):
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(g, cv2.CV_64F, 1,0)
    sy = cv2.Sobel(g, cv2.CV_64F, 0,1)
    mag = np.sqrt(sx**2 + sy**2)
    ori = np.arctan2(sy, sx)
    if mag.max() > 0:
        mag = mag / mag.max()
    h_mag, _ = np.histogram(mag, bins=16, range=(0,1), density=True)
    h_ori, _ = np.histogram(ori, bins=16, range=(-np.pi, np.pi), density=True)
    return np.concatenate([h_mag, h_ori]).astype(np.float32)

def extract_features(img):
    f1 = color_histogram(img)
    f2 = ccm_feature(img)
    f3 = sobel_feature(img)
    return np.concatenate([f1,f2,f3])[None,:]

# ===============================================================
# DETECTION
# ===============================================================
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
    h_img, w_img = th.shape
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 200:
            continue

        ratio = w / (h + 1e-9)
        if ratio < 0.2 or ratio > 5.0:
            continue

        patch = gray[y:y+h, x:x+w]
        if patch.size == 0:
            continue
        variance = np.var(patch)
        if variance < 20:
            continue

        # optionally filter very bright blobs (rocks)
        mean_v = np.mean(cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)[:,:,2])
        if mean_v > 240:
            continue

        candidates.append((c, area))

    if len(candidates) == 0:
        # fallback: slightly relax constraints (try HSV mask)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([10, 30, 20])
        upper = np.array([60, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        cnt = max(contours, key=lambda c: cv2.contourArea(c))
        return cv2.boundingRect(cnt)

    cnt = max(candidates, key=lambda x: x[1])[0]
    return cv2.boundingRect(cnt)

# ===============================================================
# SCALING
# ===============================================================
def scaling(X):
    if scaler is None: return X
    return scaler.transform(X)


# ===============================================================
# SIDEBAR MENU
# ===============================================================
menu = st.sidebar.radio("Menu", ["Home","Predict","Model Info"])

st.sidebar.markdown("---")
st.sidebar.markdown("Computer Vision Final Project — Built with SVM.")

# ===============================================================
# HOME
# ===============================================================
if menu == "Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## 🐔 Poultry Feces Classifier — Traditional ML")
    st.write("""
    Welcome!  
    This app uses **traditional features extraction + SVM** to classify poultry feces into:
    - Cocci  
    - Salmonella  
    - Healthy  

    Navigate to **Predict** to upload an image.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================================================
# PREDICT PAGE
# ===============================================================
elif menu == "Predict":
    st.markdown("## 🔍 Prediction")
    uploaded = st.file_uploader("Upload image", ["jpg","jpeg","png"])

    if uploaded:
        img_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                 caption="Input Image", width=350)
        st.markdown("</div>", unsafe_allow_html=True)

        full_feat = extract_features(img)
        full_feat = scaling(full_feat)

        probs = svm.predict_proba(full_feat)[0]
        pred_idx = np.argmax(probs)
        pred_label = svm.classes_[pred_idx]
        pred_class = label_mapping[int(pred_label)]

        with st.spinner("Detecting feces region..."):
            bbox = detect_feces_bbox(img)

        boxed = img.copy()
        crop = None

        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(boxed, (x, y), (x+w, y+h), (0, 255, 0), 3)
            crop = img[y:y+h, x:x+w]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.image(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB),
                     caption="Detected Region", width=350)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            if crop is not None:
                st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                         caption="Cropped Feces",
                         width=350)
            else:
                st.info("No feces detected.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"### 🧪 Result: **{pred_class.upper()}**")

        st.markdown("#### Probabilities:")
        for i, p in enumerate(probs):
            st.write(f"- {label_mapping[int(svm.classes_[i])]} : **{p:.4f}**")



# ===============================================================
# MODEL INFO
# ===============================================================
elif menu == "Model Info":
    st.markdown("## 📦 Model Information")
    st.write(f"- Classes: {class_names}")
    st.write(f"- SVM classes: {svm.classes_}")
    st.write(f"- Scaler: {type(scaler).__name__}")
