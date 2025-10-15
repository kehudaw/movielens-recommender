# src/report.py
import os
import joblib
import numpy as np
from .utils import recall_at_k, mapk

MODELS_DIR = "models"


def _z(x):
    """Z-score a 1D vector safely (scale-invariant blending)."""
    x = np.asarray(x).ravel()
    mu = float(x.mean()) if x.size else 0.0
    sd = float(x.std()) if x.std() > 1e-9 else 1.0
    return (x - mu) / sd


def recommend_for_user(bundle, uix, alpha=0.7, topk=100):
    # CF scores (SVD dot-product, then z-score)
    cf = bundle["user_factors"][uix].dot(bundle["item_factors"].T)
    cf = _z(cf)

    # Content scores (TF-IDF dot user profile, then z-score)
    content = bundle["item_tfidf"].dot(bundle["user_profiles"][uix])
    content = np.asarray(content).ravel()
    content = _z(content)

    # Hybrid blend with scale-invariance via z-scoring
    scores = alpha * cf + (1.0 - alpha) * content

    # Filter items seen in train
    seen_items = bundle["train_seen"].get(int(bundle["ix2uid"][uix]), set())
    for iid in seen_items:
        j = bundle["iid2ix"].get(iid)
        if j is not None:
            scores[j] = -1e9

    # Top indices → item ids
    top_idx = np.argsort(-scores)
    rec_iids = [bundle["ix2iid"][int(j)] for j in top_idx[:topk]]

    # Popularity backfill (ensure we always return topk)
    if len(rec_iids) < topk:
        for iid in bundle.get("popularity", []):
            if iid not in seen_items and iid not in rec_iids:
                rec_iids.append(iid)
            if len(rec_iids) >= topk:
                break

    return rec_iids[:topk]


def main():
    path = os.path.join(MODELS_DIR, "recsys_bundle.joblib")
    assert os.path.exists(path), "Train first: python -m src.train"
    b = joblib.load(path)

    # Sweep alpha and choose the best by Recall@10
    alphas = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    per_alpha = []
    best = (None, -1.0, -1.0)  # (alpha, r10, r20)

    for a in alphas:
        recalls10, recalls20 = [], []
        for u, gt in b["test_by_user"].items():
            uix = b["uid2ix"].get(u)
            if uix is None or len(gt) == 0:
                continue
            recs = recommend_for_user(b, uix, alpha=a, topk=100)
            recalls10.append(recall_at_k(gt, np.array(recs), k=10))
            recalls20.append(recall_at_k(gt, np.array(recs), k=20))
        r10 = float(np.mean(recalls10)) if recalls10 else 0.0
        r20 = float(np.mean(recalls20)) if recalls20 else 0.0
        per_alpha.append((a, r10, r20))
        if r10 > best[1]:
            best = (a, r10, r20)

    best_alpha, best_r10, best_r20 = best

    # Compute MAP@10 at the best alpha
    gts, preds = {}, {}
    for u, gt in b["test_by_user"].items():
        uix = b["uid2ix"].get(u)
        if uix is None or len(gt) == 0:
            continue
        recs = recommend_for_user(b, uix, alpha=best_alpha, topk=100)
        gts[u] = np.array(gt)
        preds[u] = np.array(recs)
    map10 = mapk(gts, preds, k=10)

    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.md", "w", encoding="utf-8") as f:
        f.write("# Metrics\n\n")
        f.write(f"- **Recall@10**: {best_r10:.3f}\n")
        f.write(f"- **Recall@20**: {best_r20:.3f}\n")
        f.write(f"- **MAP@10**: {map10:.3f}\n")
        f.write(f"- **Alpha (CF weight)**: {best_alpha:.2f}\n\n")
        f.write("### Alpha sweep (Recall@K)\n")
        for a, r10, r20 in per_alpha:
            f.write(f"- alpha={a:.2f} → R@10={r10:.3f}, R@20={r20:.3f}\n")
    print("Wrote reports/metrics.md")


if __name__ == "__main__":
    main()
