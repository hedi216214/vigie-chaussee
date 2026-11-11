"""
visualize.py — Figures du projet (toutes générées à partir des données / prédictions réelles).

    show_samples ............ images + masques de vérité terrain
    overlay / error_map ..... superposition masque et carte d'erreurs (vert TP, rouge FP, bleu FN)
    plot_learning_curves .... courbes perte / métrique de validation
    plot_confusion .......... matrice de confusion
    plot_roc_pr ............. courbes ROC et précision-rappel
    show_reconstruction ..... entrée / reconstruction / erreur d'un autoencodeur
    show_passability ........ chaussée, anomalie, espace libre et verdict BO3
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

COLORS = {"tp": (40, 180, 90), "fp": (220, 50, 50), "fn": (40, 110, 230), "anomaly": (242, 194, 48),
          "road": (90, 160, 255), "free": (60, 200, 120)}


def savefig(fig, path: str | None) -> None:
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")


def overlay(image: np.ndarray, mask: np.ndarray, color=COLORS["anomaly"], alpha: float = 0.5) -> np.ndarray:
    out = image.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.asarray(color, np.float32)
    return out.clip(0, 255).astype(np.uint8)


def error_map(image: np.ndarray, pred: np.ndarray, gt: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Vert = anomalie détectée (TP), rouge = fausse alarme (FP), bleu = anomalie manquée (FN)."""
    p, g = pred.astype(bool), gt.astype(bool)
    out = image.copy()
    for m, c in [(p & g, COLORS["tp"]), (p & ~g, COLORS["fp"]), (~p & g, COLORS["fn"])]:
        out = overlay(out, m, c, alpha)
    return out


def show_samples(images, masks, names=None, n: int = 6, cols: int = 3, path: str | None = None):
    n = min(n, len(images))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
    for ax, i in zip(np.ravel(axes), range(n)):
        ax.imshow(overlay(images[i], masks[i]))
        ax.set_title(names[i][:40] if names is not None else str(i), fontsize=9)
        ax.axis("off")
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.tight_layout()
    savefig(fig, path)
    return fig


def show_predictions(images, probs, gts, thr: float, names=None, n: int = 4, path: str | None = None):
    """Par ligne : image, score, masque prédit, carte d'erreurs."""
    n = min(n, len(images))
    fig, axes = plt.subplots(n, 4, figsize=(16, 2.6 * n), squeeze=False)
    for r in range(n):
        pred = probs[r] >= thr
        panels = [(images[r], "image"), (probs[r], "score d'anomalie"),
                  (overlay(images[r], pred), f"prédiction (seuil {thr:.2f})"),
                  (error_map(images[r], pred, gts[r]), "TP vert · FP rouge · FN bleu")]
        for c, (im, title) in enumerate(panels):
            axes[r, c].imshow(im, cmap="magma" if im.ndim == 2 else None, vmin=0, vmax=1 if im.ndim == 2 else None)
            axes[r, c].set_title((names[r][:28] + " — " if names is not None and c == 0 else "") + title, fontsize=8)
            axes[r, c].axis("off")
    fig.tight_layout()
    savefig(fig, path)
    return fig


def plot_learning_curves(history: dict, keys=None, title: str = "", path: str | None = None):
    keys = keys or [k for k, v in history.items() if isinstance(v, list)]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.5 * len(keys), 3.2), squeeze=False)
    for ax, k in zip(axes[0], keys):
        ax.plot(range(1, len(history[k]) + 1), history[k])
        if "best_epoch" in history:
            ax.axvline(history["best_epoch"], ls="--", c="gray", lw=1)
        ax.set_xlabel("époque")
        ax.set_title(k)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    savefig(fig, path)
    return fig


def plot_confusion(cm, labels=("normal", "anomalie"), title: str = "", path: str | None = None):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("prédit")
    ax.set_ylabel("vrai")
    ax.set_title(title)
    fig.tight_layout()
    savefig(fig, path)
    return fig


def plot_roc_pr(curves: dict, path: str | None = None, max_points: int = 2_000_000, seed: int = 42):
    """curves = {nom_modèle: (y_true, scores)} -> ROC et précision-rappel côte à côte."""
    from sklearn.metrics import precision_recall_curve, roc_curve
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    rng = np.random.default_rng(seed)
    for name, (y, s) in curves.items():
        y, s = np.asarray(y).ravel(), np.asarray(s).ravel()
        if y.size > max_points:
            idx = rng.choice(y.size, max_points, replace=False)
            y, s = y[idx], s[idx]
        fpr, tpr, _ = roc_curve(y, s)
        pr, rc, _ = precision_recall_curve(y, s)
        a1.plot(fpr, tpr, label=name)
        a2.plot(rc, pr, label=name)
    a1.plot([0, 1], [0, 1], "k--", lw=0.8)
    a1.set(xlabel="taux de faux positifs", ylabel="taux de vrais positifs", title="ROC")
    a2.set(xlabel="rappel", ylabel="précision", title="Précision-rappel")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, path)
    return fig


def show_reconstruction(inputs, recons, n: int = 6, path: str | None = None):
    """inputs / recons : (N, 3, H, W) dans [0,1]. Ligne 3 = erreur quadratique moyenne par pixel."""
    n = min(n, len(inputs))
    fig, axes = plt.subplots(3, n, figsize=(2.2 * n, 6.6), squeeze=False)
    for i in range(n):
        x, r = np.asarray(inputs[i]).transpose(1, 2, 0), np.asarray(recons[i]).transpose(1, 2, 0)
        for row, (im, t) in enumerate([(x, "entrée"), (r, "reconstruction"), (((x - r) ** 2).mean(2), "erreur")]):
            axes[row, i].imshow(im.clip(0, 1) if im.ndim == 3 else im, cmap="inferno" if im.ndim == 2 else None)
            axes[row, i].set_title(t, fontsize=8)
            axes[row, i].axis("off")
    fig.tight_layout()
    savefig(fig, path)
    return fig


def show_passability(image, road_region, anomaly, result: dict | None = None, path: str | None = None):
    """Chaussée estimée (bleu), anomalie sur la chaussée (jaune) et verdict BO3."""
    vis = overlay(image, road_region, COLORS["road"], 0.35)
    vis = overlay(vis, anomaly & road_region, COLORS["anomaly"], 0.7)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.imshow(vis)
    ax.axis("off")
    if result:
        ax.set_title(result.get("message", ""), fontsize=9, wrap=True)
    fig.tight_layout()
    savefig(fig, path)
    return fig
