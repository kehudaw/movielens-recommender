# MovieLens Recommender — Hybrid Collaborative + Content

**Goal:** Build a top‑N movie recommender using **Collaborative Filtering (SVD)** and **Content‑based TF‑IDF**, then blend them with a slider. Ships with a Streamlit app and offline Recall@K metrics.

## Highlights
- Dataset: **MovieLens 100K** (users, movies, ratings, timestamps)
- CF: Truncated SVD on the user–item matrix (implicit, thresholded on rating ≥ 4)
- Content: TF‑IDF on title + genres, user profile by liked items
- Metrics: **Recall@10/20**, **MAP@10** (holdout by timestamp per user)
- App: Streamlit — pick a user, adjust hybrid weight, get top‑N recommendations

## Quickstart (Windows)
cd "C:\Users\inson\OneDrive\Desktop\Data Science Projects\movielens-recommender"

- 1) Setup (Python 3.11 recommended)
py -3.11 -m venv .venv
call .venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt

- 2) Download MovieLens 100K
python -m src.data

- 3) Train (builds CF + content models and saves bundle)
python -m src.train

- 4) Report (writes reports\metrics.md)
python -m src.report

- 5) App
streamlit run app\streamlit_app.py

## Results
- **Recall@10:** 0.155  
- **Recall@20:** 0.189  
- **MAP@10:** 0.075  
- **Alpha (CF weight):** 0.70

### Alpha sweep
- α=0.00 → R@10=0.050, R@20=0.064
- α=0.30 → R@10=0.085, R@20=0.104
- α=0.50 → R@10=0.134, R@20=0.152
- **α=0.70 → R@10=0.155, R@20=0.189**
- α=0.90 → R@10=0.154, R@20=0.183
- α=1.00 → R@10=0.152, R@20=0.177

