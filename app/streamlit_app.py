import os, sys, joblib, numpy as np, pandas as pd, streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.report import recommend_for_user

st.set_page_config(page_title="MovieLens Recommender", page_icon="🎬", layout="wide")
st.title("🎬 MovieLens Recommender — Hybrid CF + Content")

path = os.path.join("models", "recsys_bundle.joblib")
if not os.path.exists(path):
    st.error("No model bundle found. Run `python -m src.train` (or `tasks train`) first."); st.stop()

b = joblib.load(path)

st.sidebar.header("User & blending")
user_ids = sorted(list(b["uid2ix"].keys()))
user = st.sidebar.selectbox("User ID", user_ids, index=0)
alpha = st.sidebar.slider("CF weight (alpha)", 0.0, 1.0, 0.7, 0.05)
topk = st.sidebar.slider("Top-N", 5, 50, 10, 1)

uix = b["uid2ix"][user]
recs = recommend_for_user(b, uix, alpha=alpha, topk=topk)

seen = list(b["train_seen"].get(int(user), set()))[:30]
st.subheader(f"User {user} — liked items (train)")
if not seen:
    st.caption("No liked items recorded in train split.")
else:
    st.write(", ".join([b["items_meta"][i]["title"] for i in seen if i in b["items_meta"]][:30]))

st.subheader(f"Top-{topk} recommendations (alpha={alpha:.2f})")
rows = []
for iid in recs[:topk]:
    meta = b["items_meta"].get(iid, {"title": f"Item {iid}", "genres": ""})
    rows.append({"item_id": iid, "title": meta["title"], "genres": meta["genres"]})
st.dataframe(pd.DataFrame(rows))

st.divider()
st.subheader("Search movies")
q = st.text_input("Title contains")
if q:
    hits = []
    for iid, meta in b["items_meta"].items():
        if q.lower() in meta["title"].lower():
            hits.append({"item_id": iid, "title": meta["title"], "genres": meta["genres"]})
    st.dataframe(pd.DataFrame(hits[:100]))
