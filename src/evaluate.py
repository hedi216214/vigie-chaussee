"""
evaluate.py — Métriques et protocoles d'évaluation.

    Classification (BO1)  : precision, recall, F1, accuracy, AUROC, AP (= AuPRC), FPR95, matrice de confusion
    Segmentation (BO2)    : precision, recall, F1, IoU, Dice au niveau pixel ; AUROC, AP, FPR95 sans seuil ;
                            métriques par image, par famille et par taille d'objet ; rappel par objet
Toutes les fonctions calculent les métriques à partir des prédictions réelles : aucune
valeur par défaut n'est renvoyée si les entrées sont vides (une erreur est levée).

Remarque : au niveau pixel, F1 et Dice sont mathématiquement identiques (2TP / (2TP+FP+FN)) ;
on calcule Dice à partir des sommes globales (micro) et l'on donne aussi la moyenne par image (macro).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

EPS = 1e-9


def _check(y_true, scores):
    y_true, scores = np.asarray(y_true).ravel(), np.asarray(scores, dtype=np.float64).ravel()
    if y_true.size == 0 or y_true.size != scores.size:
        raise ValueError("Entrées vides ou de tailles différentes.")
    return y_true.astype(np.uint8), scores


def fpr_at_tpr(y_true, scores, tpr_level: float = 0.95) -> float:
    """Taux de faux positifs quand on détecte `tpr_level` des positifs (FPR95 pour 0,95)."""
    y, s = _check(y_true, scores)
    fpr, tpr, _ = roc_curve(y, s)
    return float(fpr[np.searchsorted(tpr, tpr_level, side="left")])


def scores_from_confusion(tp, fp, fn, tn) -> dict:
    precision, recall = tp / (tp + fp + EPS), tp / (tp + fn + EPS)
    return {"precision": float(precision), "recall": float(recall),
            "f1": float(2 * precision * recall / (precision + recall + EPS)),
            "iou": float(tp / (tp + fp + fn + EPS)), "dice": float(2 * tp / (2 * tp + fp + fn + EPS)),
            "accuracy": float((tp + tn) / (tp + fp + fn + tn + EPS)),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def binary_classification_metrics(y_true, scores, thr: float = 0.5) -> dict:
    """Métriques de classification présence / absence (scores = probabilité ou erreur de reconstruction)."""
    y, s = _check(y_true, scores)
    pred = s >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    out = scores_from_confusion(tp, fp, fn, tn)
    both = len(np.unique(y)) == 2
    out.update(auroc=float(roc_auc_score(y, s)) if both else float("nan"),
               ap=float(average_precision_score(y, s)) if both else float("nan"),
               fpr95=fpr_at_tpr(y, s) if both else float("nan"), threshold=float(thr),
               confusion_matrix=[[tn, fp], [fn, tp]])
    return out


def best_f1_threshold(y_true, scores, n: int = 201) -> tuple[float, float]:
    """Seuil maximisant le F1 — À CALCULER SUR LA VALIDATION, puis appliquer tel quel au test."""
    y, s = _check(y_true, scores)
    grid = np.quantile(s, np.linspace(0.0, 1.0, n))
    best = (0.5, -1.0)
    for t in np.unique(grid):
        f1 = binary_classification_metrics(y, s, t)["f1"]
        if f1 > best[1]:
            best = (float(t), f1)
    return best


# --------------------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------------------
def segmentation_metrics(probs: np.ndarray, gts: np.ndarray, thr: float = 0.5) -> dict:
    """Métriques pixel agrégées sur l'ensemble des images (micro)."""
    pred, gt = np.asarray(probs) >= thr, np.asarray(gts) > 0
    if pred.shape != gt.shape:
        raise ValueError(f"Formes différentes : {pred.shape} vs {gt.shape}")
    tp = np.sum(pred & gt, dtype=np.int64)
    fp = np.sum(pred & ~gt, dtype=np.int64)
    fn = np.sum(~pred & gt, dtype=np.int64)
    tn = np.sum(~pred & ~gt, dtype=np.int64)
    return scores_from_confusion(tp, fp, fn, tn) | {"threshold": float(thr)}


def best_seg_threshold(probs: np.ndarray, gts: np.ndarray, n: int = 99) -> tuple[float, float]:
    """Seuil maximisant le Dice de VALIDATION."""
    best = (0.5, -1.0)
    for t in np.linspace(0.01, 0.99, n):
        d = segmentation_metrics(probs, gts, t)["dice"]
        if d > best[1]:
            best = (float(t), d)
    return best


def threshold_free_metrics(probs: np.ndarray, gts: np.ndarray, max_pixels: int = 3_000_000,
                           seed: int = 42) -> dict:
    """AUROC, AP (AuPRC) et FPR95 pixel (protocole OoD des benchmarks d'anomalies routières).
    Sous-échantillonnage aléatoire uniforme si trop de pixels (mémoire)."""
    y, s = np.asarray(gts).ravel() > 0, np.asarray(probs, dtype=np.float32).ravel()
    if y.size > max_pixels:
        idx = np.random.default_rng(seed).choice(y.size, max_pixels, replace=False)
        y, s = y[idx], s[idx]
    if len(np.unique(y)) < 2:
        raise ValueError("Il faut des pixels normaux ET anormaux pour l'AUROC.")
    return {"auroc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s)),
            "fpr95": fpr_at_tpr(y, s)}


def per_image_metrics(probs, gts, names, thr: float):
    """Une ligne par image : métriques, famille, aire de l'anomalie, classe de taille."""
    import pandas as pd
    from .dataset import family_of, size_bin
    rows = []
    for p, g, n in zip(probs, gts, names):
        m = segmentation_metrics(p, g, thr)
        area = int((g > 0).sum())
        rows.append({"name": n, "family": family_of(n), "gt_area_px": area, "size": size_bin(area),
                     **{k: m[k] for k in ["precision", "recall", "f1", "iou", "dice"]}})
    return pd.DataFrame(rows)


def per_group_summary(df, by: str = "family"):
    """Moyenne et écart-type par famille (ou par classe de taille) + effectif.
    Avec ~10 images de test, certaines familles ont 1 à 3 images : à interpréter avec prudence."""
    g = df.groupby(by)[["iou", "dice", "precision", "recall"]]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std"))
    out["n_images"] = df.groupby(by).size()
    return out


def object_recall(probs, gts, thr: float, min_cover: float = 0.5, names=None):
    """Rappel au niveau objet : une composante connexe de la vérité terrain est « détectée »
    si au moins `min_cover` de ses pixels sont prédits anormaux. Utile pour les petits débris
    qui pèsent peu dans l'IoU global."""
    import pandas as pd
    from scipy import ndimage
    from .dataset import family_of, size_bin
    rows = []
    names = names if names is not None else [str(i) for i in range(len(gts))]
    for p, g, n in zip(probs, gts, names):
        lab, k = ndimage.label(g > 0)
        pred = p >= thr
        for i in range(1, k + 1):
            comp = lab == i
            cover = float(pred[comp].mean())
            rows.append({"name": n, "family": family_of(n), "area_px": int(comp.sum()),
                         "size": size_bin(int(comp.sum())), "cover": cover, "detected": cover >= min_cover})
    return pd.DataFrame(rows)


def bootstrap_ci(values, n: int = 2000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    """Intervalle de confiance bootstrap de la moyenne (sur les images de test)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, (n, len(v)), replace=True).mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
