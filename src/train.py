# src/train.py
import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = "data"
ML_DIR = os.path.join(DATA_DIR, "ml-100k")
U_DATA = os.path.join(ML_DIR, "u.data")
U_ITEM = os.path.join(ML_DIR, "u.item")
MODELS_DIR = "models"


def load_raw():
    assert os.path.exists(U_DATA), "Run: tasks download"
    ratings = pd.read_csv(
        U_DATA, sep="\t", header=None,
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    items = pd.read_csv(
        U_ITEM, sep="|", header=None, encoding="latin-1",
        names=[
            "item_id", "title", "release_date", "video_release_date", "imdb_url",
            "unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
        ]
    )
    genre_cols = [
        "Action","Adventure","Animation","Childrens","Comedy","Crime","Documentary",
        "Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance",
        "Sci-Fi","Thriller","War","Western"
    ]
    items["genres"] = items[genre_cols].apply(
        lambda r: "|".join([g for g, c in zip(genre_cols, r.values) if c == 1]), axis=1
    )
    items = items[["item_id", "title", "genres"]]
    return ratings, items


def split_train_test(ratings: pd.DataFrame):
    # chronological per-user holdout (last ~20% per user if >=5 interactions)
    ratings = ratings.sort_values("timestamp")
    train_mask = np.ones(len(ratings), dtype=bool)
    test_rows = []
    for _, grp in ratings.groupby("user_id", sort=False):
        if len(grp) >= 5:
            idx = grp.index.values
            test_idx = idx[-max(1, len(grp)//5):]  # last 20% (>=1)
            test_rows.extend(list(test_idx))
    train_mask[np.isin(ratings.index.values, np.array(test_rows))] = False
    train = ratings[train_mask]
    test = ratings[~train_mask]
    return train, test


def build_mappings(ratings: pd.DataFrame):
    uids = np.sort(ratings["user_id"].unique())
    iids = np.sort(ratings["item_id"].unique())
    uid2ix = {u: i for i, u in enumerate(uids)}
    iid2ix = {i: j for j, i in enumerate(iids)}
    return uid2ix, iid2ix


def make_interaction_matrix(ratings: pd.DataFrame, uid2ix, iid2ix, thresh=4):
    # implicit positives from explicit ratings >= thresh
    pos = ratings[ratings["rating"] >= thresh]
    rows = pos["user_id"].map(uid2ix).values
    cols = pos["item_id"].map(iid2ix).values
    data = np.ones(len(pos), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(len(uid2ix), len(iid2ix)))


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    ratings, items = load_raw()
    train, test = split_train_test(ratings)
    uid2ix, iid2ix = build_mappings(train)
    ix2uid = {v: k for k, v in uid2ix.items()}
    ix2iid = {v: k for k, v in iid2ix.items()}

    X = make_interaction_matrix(train, uid2ix, iid2ix, thresh=4)

    # ---- Collaborative Filtering (SVD) ----
    # Revert to 64 comps, no L2 normalization (this scored better in your runs)
    svd = TruncatedSVD(n_components=64, random_state=42)
    user_factors = svd.fit_transform(X)      # [n_users, k]
    item_factors = svd.components_.T         # [n_items, k]

    # ---- Content-based (TF-IDF) ----
    items_sub = items[items["item_id"].isin(list(iid2ix.keys()))].copy()
    items_sub["text"] = items_sub["title"].fillna("") + " " + items_sub["genres"].fillna("")
    items_sub = items_sub.set_index("item_id").loc[sorted(iid2ix.keys())]

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english")
    item_tfidf = tfidf.fit_transform(items_sub["text"].values)  # sparse [n_items, d]

    # user content profiles = simple mean of liked items' TF-IDF (no L2)
    user_profiles = np.zeros((len(uid2ix), item_tfidf.shape[1]), dtype=np.float32)
    for u, grp in train[train["rating"] >= 4].groupby("user_id"):
        uix = uid2ix.get(u)
        if uix is None:
            continue
        liked_cols = [iid2ix[i] for i in grp["item_id"].values if i in iid2ix]
        if liked_cols:
            prof = item_tfidf[liked_cols].mean(axis=0)
            user_profiles[uix] = np.asarray(prof).ravel()

    # popularity backfill list (desc by #positive ratings)
    popularity = (
        train[train["rating"] >= 4]["item_id"]
        .value_counts()
        .index
        .tolist()
    )

    bundle = {
        "svd": svd,
        "user_factors": user_factors,
        "item_factors": item_factors,
        "tfidf": tfidf,
        "item_tfidf": item_tfidf,
        "user_profiles": user_profiles,
        "uid2ix": uid2ix,
        "iid2ix": iid2ix,
        "ix2uid": ix2uid,
        "ix2iid": ix2iid,
        "items_meta": items.set_index("item_id")[["title", "genres"]].to_dict(orient="index"),
        "train_seen": {int(u): set(g["item_id"].tolist()) for u, g in train.groupby("user_id")},
        "test_by_user": {int(u): g["item_id"].values for u, g in test.groupby("user_id")},
        "popularity": popularity,
    }
    joblib.dump(bundle, os.path.join(MODELS_DIR, "recsys_bundle.joblib"))
    print("Saved models/recsys_bundle.joblib")
    print(f"Users: {len(uid2ix)} | Items: {len(iid2ix)} | Train positives: {X.nnz}")


if __name__ == "__main__":
    train()
