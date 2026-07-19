import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import os

city = "madrid"
base_path = f"preprocessed/{city}"

os.makedirs(base_path, exist_ok=True)

# -----------------------------
# 1) Cargar particiones
# -----------------------------
train = pd.read_pickle(f"{base_path}/train.pkl")
dev = pd.read_pickle(f"{base_path}/val.pkl")
test = pd.read_pickle(f"{base_path}/test.pkl")

# Guardar tamaños originales
n_train = len(train)
n_dev = len(dev)
n_test = len(test)

# -----------------------------
# 2) Unir conjuntos
# -----------------------------
df_all = pd.concat([train, dev, test], ignore_index=True).reset_index(drop=True)

# ID único por reseña
df_all["id_img"] = df_all.index.astype("int32")

# -----------------------------
# 3) Codificación para todos
# -----------------------------
df_all["id_user"] = df_all["id_user"].astype("category").cat.codes.astype("int32")
df_all["id_restaurant"] = df_all["id_restaurant"].astype("category").cat.codes.astype("int32")

# -----------------------------
# 4) Volver a separar splits
# -----------------------------
train = df_all.iloc[:n_train].copy()
dev = df_all.iloc[n_train:n_train+n_dev].copy()
test = df_all.iloc[n_train+n_dev:].copy()

# -----------------------------
# 5) Generar embeddings
# -----------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

texts = df_all["review_full"].tolist()
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

with open(f"{base_path}/IMG_VEC", "wb") as f:
    pickle.dump(embeddings, f)

# -----------------------------
# 6) DEV_IMG (negativos desde TRAIN)
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
        "id_test": idx,
        "review_full": row["review_full"]
    })

    # NEGATIVOS (mismo restaurante en TRAIN)
    negs = train[train["id_restaurant"] == rest]

    # excluir el positivo si coincide
    negs = negs[negs["id_img"] != img]

    for _, neg in negs.iterrows():
        dev_rows.append({
            "id_user": user,
            "id_img": neg["id_img"],
            "id_restaurant": rest,
            "is_dev": 0,
            "id_test": idx,
            "review_full": neg["review_full"]
        })

DEV_IMG = pd.DataFrame(dev_rows)

# -----------------------------
# 7) TEST_IMG (negativos desde TRAIN+DEV)
# -----------------------------
train_dev_base = pd.concat([train, dev], ignore_index=True)

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
        "id_test": idx,
        "review_full": row["review_full"]
    })

    # NEGATIVOS (mismo restaurante en TRAIN+DEV)
    negs = train_dev_base[train_dev_base["id_restaurant"] == rest]

    # excluir el positivo
    negs = negs[negs["id_img"] != img]

    for _, neg in negs.iterrows():
        test_rows.append({
            "id_user": user,
            "id_img": neg["id_img"],
            "id_restaurant": rest,
            "is_dev": 0,
            "id_test": idx,
            "review_full": neg["review_full"]
        })

TEST_IMG = pd.DataFrame(test_rows)

# -----------------------------
# 8) TRAIN_IMG y TRAIN_DEV_IMG (solo positivos)
# -----------------------------
train["take"] = 1
train["id_test"] = train.index
TRAIN_IMG = train.copy()

train_dev_for_training = pd.concat([train, dev], ignore_index=True)
train_dev_for_training["take"] = 1
train_dev_for_training["id_test"] = train_dev_for_training.index
TRAIN_DEV_IMG = train_dev_for_training.copy()

# -----------------------------
# 9) Guardado
# -----------------------------
TRAIN_IMG.to_pickle(f"{base_path}/TRAIN_IMG")
DEV_IMG.to_pickle(f"{base_path}/DEV_IMG")
TEST_IMG.to_pickle(f"{base_path}/TEST_IMG")
TRAIN_DEV_IMG.to_pickle(f"{base_path}/TRAIN_DEV_IMG")

print("Todo generado correctamente.")

# -----------------------------
# 10) Verificaciones
# -----------------------------
print("\n--- Verificaciones ---")

print("Positivos por id_test (DEV):")
print(DEV_IMG.groupby("id_test")["is_dev"].sum().value_counts())

print("Positivos por id_test (TEST):")
print(TEST_IMG.groupby("id_test")["is_dev"].sum().value_counts())

print("\nTamaño candidatos TEST:")
print(TEST_IMG.groupby("id_test").size().describe())