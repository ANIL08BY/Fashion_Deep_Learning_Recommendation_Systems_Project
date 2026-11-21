# 👗 Fashion Deep Learning Recommendation System
### Moda Ürünleri İçin Görsel Tabanlı Derin Öğrenme Tavsiye Sistemi

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-ResNet50-FF6F00.svg)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-KNN-F7931E.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B.svg)](https://streamlit.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Model_Hub-FFD21E.svg)](https://huggingface.co/)

🔗 **Canlı Uygulama (Streamlit Cloud):** `https://BURAYA-STREAMLIT-LINKINIZI-YAZIN.streamlit.app`  
🤗 **Hugging Face Model Deposu:** `https://huggingface.co/Anl08By/fashion-recommender-system`  
📊 **Veri Seti:** [Kaggle - Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

---

## 📌 Proje Özeti (Overview)

Bu proje, e-ticaret ve moda platformlarında kullanılan **Görsel Tabanlı Tavsiye Sistemi (Visual Recommender System)** mimarisini uçtan uca uygular. Veri setindeki **44.441** moda ürünü görselinden **Transfer Learning (ResNet50)** ile 2048 boyutlu öznitelik vektörleri (embeddings) çıkarılmış ve **K-En Yakın Komşu (`NearestNeighbors`)** algoritması kullanılarak seçilen veya dışarıdan yüklenen bir ürüne görsel açıdan en çok benzeyen **5 ürünün** önerilmesi sağlanmıştır.

> **🇬🇧 English Summary:** An end-to-end visual fashion recommendation engine built on 44,441 product images. Feature embeddings (2048-D, L2-normalized) are extracted via a frozen ImageNet-pretrained **ResNet50 + GlobalMaxPool2D** architecture and indexed with **Scikit-Learn `NearestNeighbors`** (Euclidean distance). Model artifacts are hosted on **Hugging Face Hub**, and the interactive web UI is deployed via **Streamlit**.

---

## 🧠 Sistem Mimarisi ve Teknik Özellikler

1. **Veri Ön İşleme & EDA:**
   * Görseller `224x224x3` boyutuna ölçeklendirilip ResNet50 `preprocess_input` dönüşümünden geçirilir.
   * Windows ve Linux sunucuları arasında tam uyumluluk için tüm dosya yolları `/` standardına dönüştürülmüştür.
2. **Derin Öğrenme ile Öznitelik Çıkarımı (Feature Extraction):**
   * `ResNet50(weights = "imagenet", include_top = False, input_shape = (224, 224, 3))` temel modelinin ağırlıkları dondurulmuş (`trainable = False`) ve sonuna `GlobalMaxPool2D` katmanı eklenmiştir.
   * Çıkarılan `2048` boyutlu vektörlere **L2 Normalizasyonu** uygulanarak `(44441, 2048)` boyutlu öznitelik matrisi oluşturulmuştur.
3. **Benzerlik Modeli (`NearestNeighbors`):**
   * `NearestNeighbors(n_neighbors = 6, algorithm = "brute", metric = "euclidean")` yapılandırması ile Öklid mesafesine göre en yakın komşular hesaplanır.
   * Katalog seçimlerinde ürünün kendisi (`index 0`) otomatik atlanırken, dışarıdan yüklenen yeni görsellerde `distances[0][0] < 1e-4` kontrolü ile en yakın eşleşme korunur.
4. **Bulut Entegrasyonu (MLOps):**
   * GitHub dosya boyutu sınırlarına takılmamak için büyük model çıktıları (`Images_features.pkl`, `filenames.pkl`, `neighbors_model.pkl`, `images.zip`) **Hugging Face Model Hub** üzerinde barındırılır ve `app.py` ilk çalıştığında `hf_hub_download` ile otomatik olarak çekilir.

---

## 📂 Proje Klasör Yapısı

```text
├── app.py                              # Streamlit web arayüzü ve Hugging Face Hub entegrasyonu
├── requirements.txt                    # Canlı ortam Python bağımlılıkları
├── Fashion_Recommender_Solution-v2.ipynb # Eğitim, EDA, öznitelik çıkarımı ve MLOps Notebook'u
└── README.md                           # Proje dokümantasyonu
```
*(Not: `Images_features.pkl`, `filenames.pkl`, `neighbors_model.pkl`, `fashion_resnet50_model.keras` ve `images.zip` dosyaları boyutları nedeniyle doğrudan Hugging Face Model Hub deposunda barındırılmaktadır.)*

---

## 🚀 Yerelde Kurulum ve Çalıştırma (Local Setup)

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

```bash
# 1. Depoyu klonlayın
git clone [https://github.com/ANIL08BY/REPO-ADINIZ.git](https://github.com/ANIL08BY/REPO-ADINIZ.git)
cd REPO-ADINIZ

# 2. Sanal ortam oluşturun ve aktif edin
python -m venv venv
# Windows için:
venv\Scripts\activate
# Linux / macOS için:
source venv/bin/activate

# 3. Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt

# 4. Streamlit uygulamasını başlatın
streamlit run app.py
```
*Uygulama ilk kez çalıştığında gerekli `.pkl` dosyalarını ve `images.zip` arşivini Hugging Face üzerinden otomatik indirip çalışma dizinine çıkartacaktır.*

---

## 🛠️ Kullanılan Teknolojiler

* **Derin Öğrenme & Makine Öğrenmesi:** TensorFlow / Keras (`ResNet50`, `GlobalMaxPool2D`), Scikit-Learn (`NearestNeighbors`), NumPy
* **Veri Analizi & Görselleştirme:** Pandas, Matplotlib, Seaborn, Pillow (PIL)
* **Arayüz & Canlıya Alma (MLOps):** Streamlit, Hugging Face Hub (`huggingface_hub`)

---

## 👤 Geliştirici

**Anıl Bilgehan Yüzgeç**  
* GitHub: [@ANIL08BY](https://github.com/ANIL08BY)
