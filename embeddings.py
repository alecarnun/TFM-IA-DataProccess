import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import os

city = "barcelona"
base_path = f"preprocessed/{city}"

os.makedirs(base_path, exist_ok=True)

# -----------------------------
# 1. Cargar particiones
# -----------------------------
train = pd.read_pickle(f"{base_path}/train.pkl")
dev = pd.read_pickle(f"{base_path}/val.pkl")
test = pd.read_pickle(f"{base_path}/test.pkl")

# -----------------------------
# 2. Unir los tres conjuntos para asignar un id a cada reseña
# -----------------------------
df_all = pd.concat([train, dev, test], ignore_index=True)
df_all = df_all.reset_index(drop=True)
df_all["id_img"] = df_all.index

# -----------------------------
# 3. Repartir id_img a cada split
# -----------------------------
train["id_img"] = df_all.loc[:len(train)-1, "id_img"].values
dev["id_img"] = df_all.loc[len(train):len(train)+len(dev)-1, "id_img"].values
test["id_img"] = df_all.loc[len(train)+len(dev):, "id_img"].values
test["id_test"] = test["id_img"]

# -----------------------------
# 4. Generar embeddings con MiniLM
# -----------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

texts = df_all["review_full"].tolist()
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# -----------------------------
# 5. Guardado
# -----------------------------
with open(f"{base_path}/IMG_VEC", "wb") as f:
    pickle.dump(embeddings, f)

train = train.reset_index(drop=True)
dev = dev.reset_index(drop=True)
test = test.reset_index(drop=True)

train["take"] = 1
dev["is_dev"] = 1
test["is_dev"] = 1

train.to_pickle(f"{base_path}/TRAIN_IMG")
dev.to_pickle(f"{base_path}/DEV_IMG")
test.to_pickle(f"{base_path}/TEST_IMG")

train_dev = pd.concat([train, dev], ignore_index=True)
train_dev["take"] = 1
train_dev.to_pickle(f"{base_path}/TRAIN_DEV_IMG")

print("Todo generado correctamente en:", base_path)
print("Archivos creados: TRAIN_IMG, DEV_IMG, TEST_IMG, TRAIN_DEV_IMG, IMG_VEC")