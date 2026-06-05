import pandas as pd
from matplotlib import pyplot as plt

# =========================
# CONFIG
# =========================
city = "madrid"
csv_path = f"{city}.csv"
base_path = f"preprocessed/{city}"

# =========================
# 1. CARGAR CSV ORIGINAL
# =========================
df = pd.read_csv(csv_path, low_memory=False)

df["id_user"] = df["user_id"].astype(str).str.split("_").str[1].str[:-4]

df = df[df["review_full"].notnull() & (df["review_full"].astype(str).str.len() > 0)]

df["rating_review"] = pd.to_numeric(df["rating_review"], errors="coerce")
df = df.dropna(subset=["rating_review"])

df = df[df["rating_review"].isin([4, 5])]

df = df.sort_values("date").drop_duplicates(
    subset=["id_user", "url_restaurant"], keep="last"
)

# RESTAURANTES
df["id_restaurant"] = df["restaurant_name"].astype("category").cat.codes

# =========================
# 2. CARGAR SPLITS
# =========================
train = pd.read_pickle(f"{base_path}/train.pkl")
val = pd.read_pickle(f"{base_path}/val.pkl")
test = pd.read_pickle(f"{base_path}/test.pkl")

# =========================
# 3. FUNCIÓN DE MÉTRICAS
# =========================
def stats(df):
    return {
        "reviews": len(df),
        "users": df["id_user"].nunique(),
        "restaurants": df["id_restaurant"].nunique()
    }

# =========================
# 4. CALCULAR TODO
# =========================
original_stats = stats(df)
train_stats = stats(train)
val_stats = stats(val)
test_stats = stats(test)

total_split_reviews = train_stats["reviews"] + val_stats["reviews"] + test_stats["reviews"]

# =========================
# 5. PRINT CLARO
# =========================
print("\n===== ORIGINAL DATA =====")
print(original_stats)

print("\n===== SPLITS =====")
print("Train:", train_stats)
print("Val:", val_stats)
print("Test:", test_stats)

print("\n===== COVERAGE =====")
print(f"Train %: {100 * train_stats['reviews'] / original_stats['reviews']:.2f}%")
print(f"Val %:   {100 * val_stats['reviews'] / original_stats['reviews']:.2f}%")
print(f"Test %:  {100 * test_stats['reviews'] / original_stats['reviews']:.2f}%")

print("\n===== CONSISTENCY CHECK =====")
print(f"Total split reviews: {total_split_reviews}")
print(f"Original reviews:    {original_stats['reviews']}")

# =========================
# 6. DISPERSIÓN (SOLO TRAIN)
# =========================
user_counts = train["id_user"].value_counts()

print("\n===== TRAIN USER ACTIVITY =====")
print(f"Min reviews/user: {user_counts.min()}")
print(f"Max reviews/user: {user_counts.max()}")
print(f"Mean: {user_counts.mean():.2f}")
print(f"Median: {user_counts.median()}")

print("\nUsers with 1 review:", (user_counts == 1).sum())
print("Users with >=2 reviews:", (user_counts >= 2).sum())

plt.figure(figsize=(8,4))

plt.hist(user_counts[user_counts <= 50], bins=25)

plt.xticks(range(0, 51, 5))
plt.title("Reviews per user distribution (Train set)")
plt.xlabel("Number of reviews")
plt.ylabel("Number of users")

plt.tight_layout()
plt.savefig(f"{city}_train_reviews_per_user.png", dpi=300, bbox_inches="tight")
plt.close()