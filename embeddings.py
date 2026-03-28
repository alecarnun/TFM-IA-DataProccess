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
df_all = pd.concat([train, dev, test], ignore_index=True).reset_index(drop=True)
df_all["id_img"] = df_all.index

# -----------------------------
# 3. Repartir id_img a cada split  ← NECESARIO
# -----------------------------
train["id_img"] = df_all.loc[:len(train)-1, "id_img"].values
dev["id_img"] = df_all.loc[len(train):len(train)+len(dev)-1, "id_img"].values
test["id_img"] = df_all.loc[len(train)+len(dev):, "id_img"].values

# -----------------------------
# 3. Codificar usuarios e imágenes
# -----------------------------
for df in [train, dev, test]:
    df["id_user"] = df["id_user"].astype("category").cat.codes.astype("int32")
    df["id_img"] = df["id_img"].astype("category").cat.codes.astype("int32")
    df["id_restaurant"] = df["id_restaurant"].astype("category").cat.codes.astype("int32")

# -----------------------------
# 4. Generar embeddings con MiniLM
# -----------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

texts = df_all["review_full"].tolist()
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

with open(f"{base_path}/IMG_VEC", "wb") as f:
    pickle.dump(embeddings, f)

# -----------------------------
# 5. Construir DEV_IMG con negativos
# -----------------------------
dev_rows = []

for idx, row in dev.iterrows():
    user = row["id_user"]
    rest = row["id_restaurant"]
    img = row["id_img"]

    # POSITIVO
    dev_rows.append({
        "id_user": user,
        "id_img": img,
        "id_restaurant": rest,
        "is_dev": 1,
        "id_test": idx
    })

    # NEGATIVOS: todas las fotos del mismo restaurante en TRAIN
    negs = train[train["id_restaurant"] == rest]

    for _, neg in negs.iterrows():
        dev_rows.append({
            "id_user": user,
            "id_img": neg["id_img"],
            "id_restaurant": rest,
            "is_dev": 0,
            "id_test": idx
        })

DEV_IMG = pd.DataFrame(dev_rows)

# -----------------------------
# 6. Construir TEST_IMG igual que DEV_IMG
# -----------------------------
train_dev_base = pd.concat([train, dev], ignore_index=True).reset_index(drop=True)
train_dev_base["id_img"] = train_dev_base["id_img"].astype("int32")
train_dev_base["id_restaurant"] = train_dev_base["id_restaurant"].astype("int32")

test_rows = []

for idx, row in test.iterrows():
    user = row["id_user"]
    rest = row["id_restaurant"]
    img = row["id_img"]

    # POSITIVO
    test_rows.append({
        "id_user": user,
        "id_img": img,
        "id_restaurant": rest,
        "is_dev": 1,
        "id_test": idx
    })

    # NEGATIVOS
    negs = train_dev_base[train_dev_base["id_restaurant"] == rest]

    for _, neg in negs.iterrows():
        test_rows.append({
            "id_user": user,
            "id_img": neg["id_img"],
            "id_restaurant": rest,
            "is_dev": 0,
            "id_test": idx
        })

TEST_IMG = pd.DataFrame(test_rows)

# -----------------------------
# 7. TRAIN_IMG y TRAIN_DEV_IMG
# -----------------------------
train["take"] = 1
train["id_test"] = train.index
TRAIN_IMG = train.copy()

train_dev_for_training = pd.concat([train, dev], ignore_index=True)
train_dev_for_training["take"] = 1
train_dev_for_training["id_test"] = train_dev_for_training.index
TRAIN_DEV_IMG = train_dev_for_training.copy()

# -----------------------------
# Añadir número de imágenes por test case
# -----------------------------
DEV_IMG["testcase_num_images"] = DEV_IMG.groupby("id_test")["id_img"].transform("count")
TEST_IMG["testcase_num_images"] = TEST_IMG.groupby("id_test")["id_img"].transform("count")

# -----------------------------
# 8. Guardado final
# -----------------------------
TRAIN_IMG.to_pickle(f"{base_path}/TRAIN_IMG")
DEV_IMG.to_pickle(f"{base_path}/DEV_IMG")
TEST_IMG.to_pickle(f"{base_path}/TEST_IMG")
TRAIN_DEV_IMG.to_pickle(f"{base_path}/TRAIN_DEV_IMG")

print("Todo generado correctamente.")