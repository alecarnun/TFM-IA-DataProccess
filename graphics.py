import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Cargar PKL
# -----------------------------
train = pd.read_pickle("preprocessed/barcelona/train.pkl")
test = pd.read_pickle("preprocessed/barcelona/test.pkl")

# -----------------------------
# Combinar para estadísticas globales
# -----------------------------
combined = pd.concat([train, test], ignore_index=True)

# -----------------------------
# Distribución de reseñas por usuario
# -----------------------------
user_counts = combined["user_id"].value_counts()

sns.histplot(user_counts[user_counts <= 50], bins=50)
plt.title("Distribución de reseñas por usuario (Train+Test)")
plt.xlabel("Número de reseñas por usuario")
plt.ylabel("Número de usuarios")
plt.show()

# -----------------------------
# Media y mediana
# -----------------------------
mean_reviews = user_counts.mean()
median_reviews = user_counts.median()

print(f"Media de reseñas por usuario: {mean_reviews:.2f}")
print(f"Mediana de reseñas por usuario: {median_reviews}")

# -----------------------------
# Longitud de reseñas
# -----------------------------
train["review_length"] = train["review_full"].str.len()

plt.scatter(train["review_length"], train["rating_review"], alpha=0.3)
plt.xlabel("Longitud de la reseña")
plt.ylabel("Rating")
plt.title("Relación entre longitud de reseña y rating")
plt.show()

# -----------------------------
# Distribución de longitudes
# -----------------------------
sns.kdeplot(train["review_full"].str.len(), fill=True)
plt.title("Distribución de longitudes de reseñas")
plt.xlabel("Longitud (caracteres)")
plt.show()