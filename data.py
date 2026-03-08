import pandas as pd
import os
from langdetect import detect, LangDetectException
from sklearn.model_selection import train_test_split

csv_path = "barcelona.csv"
city_name = "barcelona"

# ============================================================
# 1. Cargar CSV original (NO se modifica)
# ============================================================
df = pd.read_csv(csv_path, sep=",", quotechar='"', engine="python")

# ============================================================
# 2. Extraer id_user sin tocar el CSV original
# ============================================================
df["id_user"] = df["user_id"].str.split("_").str[1].str[:-4]

# ============================================================
# 3. Filtrar reseñas válidas
# ============================================================
df = df[df["review_full"].notnull() & (df["review_full"].astype(str).str.len() > 0)]

# ============================================================
# 4. Filtrar solo reseñas en inglés
# ============================================================
def is_english(text):
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False

df = df[df["review_full"].apply(is_english)]

# ============================================================
# 5. Una reseña por restaurante por usuario
# ============================================================
df = df.sort_values("date").drop_duplicates(
    subset=["id_user", "url_restaurant"], keep="last"
)

# ============================================================
# 6. Filtrar ratings 4 y 5
# ============================================================
df = df[df["rating_review"].isin([4, 5])]

# ============================================================
# 7. Separar usuarios según nº reseñas
# ============================================================
user_counts = df["id_user"].value_counts()
users_1 = user_counts[user_counts == 1].index
users_2plus = user_counts[user_counts >= 2].index

df_1 = df[df["id_user"].isin(users_1)]
df_2plus = df[df["id_user"].isin(users_2plus)]

# ============================================================
# 8. Crear train/test para usuarios con ≥2 reseñas
# ============================================================
train_rows = []
test_rows = []

for user, group in df_2plus.groupby("id_user"):
    group = group.sort_values("date")
    test_rows.append(group.iloc[-1].to_dict())
    train_rows.extend(group.iloc[:-1].to_dict("records"))

train_df = pd.DataFrame(train_rows)
test_df = pd.DataFrame(test_rows)

# ============================================================
# 9. Añadir usuarios con 1 reseña al train
# ============================================================
train_df = pd.concat([train_df, df_1], ignore_index=True)

# ============================================================
# 10. Crear validation desde train
# ============================================================
train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42)

# ============================================================
# 11. Crear id_restaurant SOLO en los splits finales
# ============================================================
for split in [train_df, val_df, test_df]:
    split["id_restaurant"] = split["restaurant_name"].astype("category").cat.codes

# ============================================================
# 12. Seleccionar solo columnas necesarias para BRIE
# ============================================================
cols_to_keep = [
    "id_user",
    "id_restaurant",
    "review_full",
    "rating_review",
]

train_df = train_df[cols_to_keep]
val_df = val_df[cols_to_keep]
test_df = test_df[cols_to_keep]

# ============================================================
# 13. Guardar PKL limpios
# ============================================================
os.makedirs(f"preprocessed/{city_name}", exist_ok=True)

train_df.to_pickle(f"preprocessed/{city_name}/train.pkl")
val_df.to_pickle(f"preprocessed/{city_name}/val.pkl")
test_df.to_pickle(f"preprocessed/{city_name}/test.pkl")

print(f"Procesamiento de {city_name} completado.")
print(f"Train: {len(train_df)} filas | Val: {len(val_df)} filas | Test: {len(test_df)} filas")