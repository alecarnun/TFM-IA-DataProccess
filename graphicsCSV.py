import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Cargar CSV original
# -----------------------------
df = pd.read_csv("tokyo.csv", low_memory=False)

# -----------------------------
# Limpieza mínima necesaria
# -----------------------------

# 1. Limpiar user_id (igual que en el preprocesado)
df["user_id"] = df["user_id"].astype(str).str.split("_").str[1].str[:-4]

# 2. Eliminar reseñas vacías
df = df[df["review_full"].notnull() & (df["review_full"].astype(str).str.len() > 0)]

# 3. Convertir rating_review a numérico
df["rating_review"] = pd.to_numeric(df["rating_review"], errors="coerce")
df = df.dropna(subset=["rating_review"])

# 4. Calcular longitud de reseña
df["review_length"] = df["review_full"].astype(str).str.len()

# -----------------------------
# 1. Distribución de reseñas por usuario
# -----------------------------
user_counts = df["user_id"].value_counts()

sns.histplot(user_counts[user_counts <= 20], bins=20)
plt.title("Distribución de reseñas por usuario")
plt.xlabel("Número de reseñas por usuario")
plt.ylabel("Número de usuarios")
plt.show()

print(f"Media de reseñas por usuario: {user_counts.mean():.2f}")
print(f"Mediana de reseñas por usuario: {user_counts.median()}")

# -----------------------------
# 2. Relación entre longitud de reseña y rating
# -----------------------------
plt.scatter(df["review_length"], df["rating_review"], alpha=0.3)
plt.xlabel("Longitud de la reseña")
plt.ylabel("Rating")
plt.title("Relación entre longitud de reseña y rating")
plt.show()

# -----------------------------
# 3. Distribución de longitudes de reseñas
# -----------------------------
sns.kdeplot(df["review_length"], fill=True)
plt.title("Distribución de longitudes de reseñas")
plt.xlabel("Longitud (caracteres)")
plt.show()