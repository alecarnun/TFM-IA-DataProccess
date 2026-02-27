import pandas as pd
import os
from langdetect import detect, LangDetectException
from sklearn.model_selection import train_test_split

csv_path = "barcelona.csv"
city_name = "barcelona"

df = pd.read_csv(csv_path, sep=",", quotechar='"', engine="python")

# -----------------------------
# Extraer el User Id
# -----------------------------
df["user_id"] = df["user_id"].str.split("_").str[1].str[:-4]

# -----------------------------
# Filtrar las reseñas válidas
# -----------------------------
df = df[df["review_full"].notnull() & (df["review_full"].astype(str).str.len() > 0)]

# -----------------------------
# Filtrar solo reseñas en inglés
# -----------------------------
def is_english(text):
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False

df = df[df["review_full"].apply(is_english)]

# -----------------------------
# Una reseña por restaurante
# -----------------------------
df = df.sort_values("date").drop_duplicates(
    subset=["user_id", "url_restaurant"], keep="last"
)

# -----------------------------
# Filtrar ratings 4 y 5
# -----------------------------
df = df[df["rating_review"].isin([4, 5])]

# -----------------------------
# Separar usuarios según nº reseñas
# -----------------------------
user_counts = df["user_id"].value_counts()
users_1 = user_counts[user_counts == 1].index
users_2plus = user_counts[user_counts >= 2].index

df_1 = df[df["user_id"].isin(users_1)]
df_2plus = df[df["user_id"].isin(users_2plus)]

# -----------------------------
# Crear train/test para usuarios con ≥2 reseñas
# -----------------------------
train_rows = []
test_rows = []

for user, group in df_2plus.groupby("user_id"):
    group = group.sort_values("date")
    test_rows.append(group.iloc[-1].to_dict())
    train_rows.extend(group.iloc[:-1].to_dict("records"))

train_df = pd.DataFrame(train_rows)
test_df = pd.DataFrame(test_rows)

# -----------------------------
# Añadir usuarios con 1 reseña al train
# -----------------------------
train_df = pd.concat([train_df, df_1], ignore_index=True)

# -----------------------------
# Crear validation desde train
# -----------------------------
train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42)

# -----------------------------
# LIMPIAR SALTOS DE LÍNEA AQUÍ
# -----------------------------
def clean_text(x):
    if isinstance(x, str):
        return x.replace("\n", " ").replace("\r", " ")
    return x

for df_tmp in [train_df, val_df, test_df]:
    for col in df_tmp.select_dtypes(include=["object"]).columns:
        df_tmp[col] = df_tmp[col].map(clean_text)

# -----------------------------
# Guardar CSVs limpios
# -----------------------------

os.makedirs(f"preprocessed/{city_name}", exist_ok=True)

train_df.to_pickle(f"preprocessed/{city_name}/train.pkl")
val_df.to_pickle(f"preprocessed/{city_name}/val.pkl")
test_df.to_pickle(f"preprocessed/{city_name}/test.pkl")

print(f"Procesamiento de {city_name} completado.")
print(f"Train: {len(train_df)} filas | Test: {len(test_df)} filas")