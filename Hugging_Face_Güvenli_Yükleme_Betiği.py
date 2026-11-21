import os
import shutil
from dotenv import load_dotenv
from huggingface_hub import HfApi

# .env dosyasındaki gizli değişkenleri yükle
load_dotenv()

hf_token = os.getenv("HF_TOKEN")
repo_id = os.getenv("HF_REPO_ID")

if not hf_token or "BURAYA" in hf_token:
    raise ValueError("Lütfen önce .env dosyasındaki HF_TOKEN ve HF_REPO_ID alanlarını kendi bilgilerinizle güncelleyin.")

# 1. images/ klasörünü otomatik olarak images.zip haline getir (yoksa oluşturur)
if not os.path.exists("images.zip") and os.path.exists("images"):
    print("images.zip oluşturuluyor, lütfen bekleyin...")
    shutil.make_archive("images_archive", "zip", root_dir=".", base_dir="images")
    os.rename("images_archive.zip", "images.zip")

# 2. Hugging Face Model Hub'a güvenli yükleme yap
api = HfApi(token=hf_token)
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

upload_list = [
    "app.py",
    "requirements.txt",
    "Images_features.pkl",
    "filenames.pkl",
    "neighbors_model.pkl",
    "fashion_resnet50_model.keras",
    "images.zip",
]

for filename in upload_list:
    if os.path.exists(filename):
        print(f"Yükleniyor: {filename} ...")
        api.upload_file(
            path_or_fileobj=filename,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model",
        )

print(f"✅ Yükleme tamamlandı! Model bağlantınız: https://huggingface.co/{repo_id}")