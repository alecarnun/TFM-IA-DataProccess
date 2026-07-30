import os
import random
from collections import Counter

import pandas as pd
from langdetect import detect, LangDetectException

# ==========================================================
# Ficheros
# ==========================================================

INPUT_CSV = "madrid.csv"
OUTPUT_CITY = "reduced_madrid"

SEED = 42
random.seed(SEED)

# Estadísticas Objetivo: Delhi
TARGET_USERS = 36088

# ==========================================================
# Carga de datos
# ==========================================================

print("=" * 60)
print("Loading Madrid dataset...")
print("=" * 60)

df = pd.read_csv(
    INPUT_CSV,
    sep=",",
    quotechar='"',
    engine="python"
)

print(f"Original reviews : {len(df):,}")

df["id_user"] = (
    df["user_id"]
    .str.split("_")
    .str[1]
    .str[:-4]
)

df = df[df["review_full"].notna()]
df = df[df["review_full"].astype(str).str.strip() != ""]

def is_english(text):
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False

print("Filtering English reviews...")

df = df[df["review_full"].apply(is_english)]

df = (
    df.sort_values("date")
    .drop_duplicates(
        subset=["id_user", "url_restaurant"],
        keep="last"
    )
)

df = df[df["rating_review"].isin([4, 5])].reset_index(drop=True)

print()
print("=" * 60)
print("Madrid after preprocessing")
print("=" * 60)

print(f"Users       : {df['id_user'].nunique():,}")
print(f"Restaurants : {df['url_restaurant'].nunique():,}")
print(f"Reviews     : {len(df):,}")

reviews_per_user = df.groupby("id_user").size()

distribution = Counter(reviews_per_user.values)

print("\nDistribution of reviews per user:\n")

for n in sorted(distribution):
    print(f"{n:3d} reviews : {distribution[n]:8,d} users")

print()
print("=" * 60)
print("Randomly selecting users...")
print("=" * 60)

reviews_per_user = df.groupby("id_user").size()

distribution = Counter(reviews_per_user.values)

selected_users = []

total_users = reviews_per_user.shape[0]

for n_reviews in sorted(distribution):

    users = reviews_per_user[
        reviews_per_user == n_reviews
        ].index.tolist()

    random.shuffle(users)

    proportion = distribution[n_reviews] / total_users

    n_select = round(proportion * TARGET_USERS)

    selected_users.extend(users[:n_select])

# Ajuste por redondeos
if len(selected_users) > TARGET_USERS:
    selected_users = random.sample(selected_users, TARGET_USERS)

elif len(selected_users) < TARGET_USERS:

    remaining = list(
        set(df["id_user"]) - set(selected_users)
    )

    selected_users.extend(
        random.sample(
            remaining,
            TARGET_USERS - len(selected_users)
        )
    )

reduced_df = df[
    df["id_user"].isin(selected_users)
].copy()

# Estadísticas del dataset reducido
reduced_reviews_per_user = (
    reduced_df
    .groupby("id_user")
    .size()
)

print()

print("=" * 60)
print("Reduced dataset")
print("=" * 60)

print(f"Users       : {reduced_df['id_user'].nunique():,}")
print(f"Restaurants : {reduced_df['url_restaurant'].nunique():,}")
print(f"Reviews     : {len(reduced_df):,}")

print()

print(
    "Average reviews/user :",
    round(len(reduced_df) / reduced_df["id_user"].nunique(), 2)
)

# ==========================================================
# TRAIN / VALIDATION / TEST SPLIT
# ==========================================================

print()
print("=" * 60)
print("Creating train/validation/test split...")
print("=" * 60)

df = reduced_df.copy()

# ----------------------------------------------------------
# Separar usuarios según número de reviews
# ----------------------------------------------------------

user_counts = df["id_user"].value_counts()

users_1 = user_counts[user_counts == 1].index
users_2plus = user_counts[user_counts >= 2].index

df_1 = df[df["id_user"].isin(users_1)]
df_2plus = df[df["id_user"].isin(users_2plus)]

# ----------------------------------------------------------
# TRAIN / TEST
# ----------------------------------------------------------

train_rows = []
test_rows = []

for user, group in df_2plus.groupby("id_user"):

    group = group.sort_values("date")

    # última review -> test
    test_rows.append(group.iloc[-1].to_dict())

    # resto -> train
    train_rows.extend(group.iloc[:-1].to_dict("records"))

train_df = pd.DataFrame(train_rows)
test_df = pd.DataFrame(test_rows)

# ----------------------------------------------------------
# Usuarios con una única review permanecen en train
# ----------------------------------------------------------

train_df = pd.concat(
    [train_df, df_1],
    ignore_index=True
)

# ----------------------------------------------------------
# TRAIN / VALIDATION
# ----------------------------------------------------------

train_rows = []
val_rows = []

for user, group in train_df.groupby("id_user"):

    group = group.sort_values("date")

    if len(group) >= 2:

        # última -> validation
        val_rows.append(group.iloc[-1].to_dict())

        # resto -> train
        train_rows.extend(group.iloc[:-1].to_dict("records"))

    else:

        train_rows.extend(group.to_dict("records"))

train_df = pd.DataFrame(train_rows)
val_df = pd.DataFrame(val_rows)

# ----------------------------------------------------------
# Codificar restaurantes
# ----------------------------------------------------------

all_restaurants = pd.concat([
    train_df["restaurant_name"],
    val_df["restaurant_name"],
    test_df["restaurant_name"]
]).astype("category")

restaurant_codes = dict(
    zip(
        all_restaurants.cat.categories,
        range(len(all_restaurants.cat.categories))
    )
)

for split in [train_df, val_df, test_df]:

    split["id_restaurant"] = (
        split["restaurant_name"]
        .map(restaurant_codes)
        .astype("int32")
    )

# ----------------------------------------------------------
# Columnas utilizadas por BRIE
# ----------------------------------------------------------

cols_to_keep = [
    "id_user",
    "id_restaurant",
    "review_full",
    "rating_review",
]

train_df = train_df[cols_to_keep]
val_df = val_df[cols_to_keep]
test_df = test_df[cols_to_keep]

# ----------------------------------------------------------
# Guardar
# ----------------------------------------------------------

output_path = f"preprocessed/{OUTPUT_CITY}"

train_df.to_pickle(f"{output_path}/train.pkl")
val_df.to_pickle(f"{output_path}/val.pkl")
test_df.to_pickle(f"{output_path}/test.pkl")

print()

print("Files saved in:")
print(output_path)

print()

print("Train :", len(train_df))
print("Val   :", len(val_df))
print("Test  :", len(test_df))

print()

print("Users in train :", train_df["id_user"].nunique())
print("Users in val   :", val_df["id_user"].nunique())
print("Users in test  :", test_df["id_user"].nunique())

print()
print("=" * 60)
print("Final statistics")
print("=" * 60)

print(f"Reviews remaining     : {len(reduced_df):,}")

print()

print("Split sizes")
print(f"Train : {len(train_df):,}")
print(f"Val   : {len(val_df):,}")
print(f"Test  : {len(test_df):,}")

print()

print("Unique users")
print(f"Train : {train_df['id_user'].nunique():,}")
print(f"Val   : {val_df['id_user'].nunique():,}")
print(f"Test  : {test_df['id_user'].nunique():,}")

print()

print("Unique restaurants")
print(f"Train : {train_df['id_restaurant'].nunique():,}")
print(f"Val   : {val_df['id_restaurant'].nunique():,}")
print(f"Test  : {test_df['id_restaurant'].nunique():,}")

print()

print("Reviews per split (%)")

total = len(train_df) + len(val_df) + len(test_df)

print(f"Train : {100*len(train_df)/total:.2f}%")
print(f"Val   : {100*len(val_df)/total:.2f}%")
print(f"Test  : {100*len(test_df)/total:.2f}%")

print("Users with one review:",
      (reduced_reviews_per_user == 1).sum())

print("Users with >=2 reviews:",
      (reduced_reviews_per_user >= 2).sum())

print("Median reviews/user:",
      reduced_reviews_per_user.median())

print("Mean reviews/user:",
      reduced_reviews_per_user.mean())

print("Max reviews/user:",
      reduced_reviews_per_user.max())

print("\nRestaurants:", df["url_restaurant"].nunique())

print()

print("Distribution after reduction:")

dist = reduced_reviews_per_user.value_counts().sort_index()

for n_reviews, n_users in dist.items():
    print(f"{n_reviews:2d} reviews : {n_users:6,d} users")

print()

print("=" * 60)
print("Done!")
print("=" * 60)