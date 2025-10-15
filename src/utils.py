import numpy as np

def recall_at_k(gt_items: np.ndarray, rec_items: np.ndarray, k: int = 10) -> float:
    if gt_items.size == 0:
        return 0.0
    hits = len(set(gt_items) & set(rec_items[:k]))
    return hits / float(min(k, len(gt_items)))

def apk(actual: np.ndarray, predicted: np.ndarray, k: int = 10) -> float:
    if actual.size == 0:
        return 0.0
    score = 0.0; hits = 0
    for i, p in enumerate(predicted[:k]):
        if p in actual and p not in predicted[:i]:
            hits += 1
            score += hits / float(i + 1)
    return score / min(len(actual), k)

def mapk(ground_truth, predictions, k: int = 10) -> float:
    vals = []
    for u in ground_truth:
        vals.append(apk(ground_truth[u], predictions.get(u, np.array([])), k=k))
    return float(np.mean(vals)) if vals else 0.0
