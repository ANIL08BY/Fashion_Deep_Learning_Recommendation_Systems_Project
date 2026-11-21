import os
import zipfile
import pickle as pkl
import numpy as np
from numpy.linalg import norm
from PIL import Image
import streamlit as st
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title = "Fashion Recommender System", page_icon="👗", layout = "wide")
st.title("👗 Fashion Deep Learning Recommendation System")
st.write("Katalogdan bir ürün seçin veya yeni bir görsel yükleyin; **ResNet50 + NearestNeighbors** modeli en benzer 5 ürünü önersin.")

REPO_ID = "Anl08By/fashion-recommender-system"

@st.cache_resource
def load_resources():
    # Dosyalar yerelde yoksa (ör. Streamlit Cloud'da) Hugging Face Model Hub'dan ücretsiz indir
    for fname in ["Images_features.pkl", "filenames.pkl", "neighbors_model.pkl", "images.zip"]:
        if not os.path.exists(fname):
            hf_hub_download(repo_id=REPO_ID, filename=fname, repo_type="model", local_dir=".")

    if not os.path.exists("images") and os.path.exists("images.zip"):
        with zipfile.ZipFile("images.zip", "r") as z:
            z.extractall(".")

    features = np.array(pkl.load(open("Images_features.pkl", "rb")))
    files = [f.replace("\\", "/") for f in pkl.load(open("filenames.pkl", "rb"))]

    if os.path.exists("fashion_resnet50_model.keras"):
        cnn_model = load_model("fashion_resnet50_model.keras")
    else:
        base_model = ResNet50(weights = "imagenet", include_top = False, input_shape = (224, 224, 3))
        base_model.trainable = False
        cnn_model = Sequential([base_model, GlobalMaxPool2D()])

    if os.path.exists("neighbors_model.pkl"):
        knn = pkl.load(open("neighbors_model.pkl", "rb"))
    else:
        knn = NearestNeighbors(n_neighbors = 6, algorithm = "brute", metric = "euclidean")
        knn.fit(features)

    return cnn_model, knn, files, features

def extract_features_pil(img_pil, cnn_model):
    img_resized = img_pil.convert("RGB").resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_expand = np.expand_dims(img_array, axis = 0)
    result = cnn_model.predict(preprocess_input(img_expand), verbose = 0).flatten()
    return result / norm(result)

model, neighbors, filenames, Image_features = load_resources()

tab1, tab2 = st.tabs(["📂 Katalogdan Ürün Seç", "📤 Yeni Görsel Yükle"])

with tab1:
    selected_file = st.selectbox("Katalogdan bir ürün seçin (İlk 1.000 ürün):", filenames[:1000])
    if st.button("Benzer Ürünleri Öner", key = "btn_catalog"):
        idx = filenames.index(selected_file)
        distances, indices = neighbors.kneighbors([Image_features[idx]])

        if os.path.exists(selected_file):
            st.subheader("Seçilen Ürün")
            st.image(selected_file, width = 200)

        st.subheader("Önerilen En Benzer 5 Ürün")
        cols = st.columns(5)
        for i, col in enumerate(cols, start = 1):
            rec_path = filenames[indices[0][i]]
            with col:
                if os.path.exists(rec_path):
                    st.image(rec_path, use_container_width = True)
                st.caption(f"{os.path.basename(rec_path)} (Mesafe: {distances[0][i]:.2f})")

with tab2:
    uploaded_file = st.file_uploader("Bir moda ürünü görseli yükleyin (.jpg, .jpeg, .png):", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        query_pil = Image.open(uploaded_file)
        st.subheader("Yüklenen Ürün")
        st.image(query_pil, width = 200)

        input_vec = extract_features_pil(query_pil, model)
        distances, indices = neighbors.kneighbors([input_vec])
        start = 1 if distances[0][0] < 1e-4 else 0

        st.subheader("Önerilen En Benzer 5 Ürün")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            rec_idx = indices[0][start + i]
            rec_path = filenames[rec_idx]
            rec_dist = distances[0][start + i]
            with col:
                if os.path.exists(rec_path):
                    st.image(rec_path, use_container_width=True)
                st.caption(f"{os.path.basename(rec_path)} (Mesafe: {rec_dist:.2f})")
