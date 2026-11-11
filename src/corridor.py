"""
corridor.py — BO3 : couloir de conduite fixe, cibles d'emprise et classes de passabilité.

RoadAnomaly n'annote pas la chaussée et ne donne aucune calibration caméra. BO3 est donc défini
sur un COULOIR DE CONDUITE FIXE : un trapèze devant le véhicule, du bas de l'image vers l'horizon
supposé (hypothèse de travail, identique pour toutes les images, à discuter dans le rapport).

Pour chaque image, trois cibles sont calculées à partir du VRAI masque d'anomalie :
    occupation   = part du couloir couverte par l'anomalie
    libre_gauche = espace libre à gauche de l'obstacle, en fraction de la largeur du couloir
    libre_droite = idem à droite
(1,0 et 1,0 quand aucun obstacle ne touche le couloir.) Les modèles BO3 apprennent à prédire ces
trois nombres directement depuis l'image ; les classes d'occupation et de passabilité s'en
déduisent par les mêmes règles (src/passability.py) pour la vérité terrain et pour les prédictions.
"""
from __future__ import annotations

import numpy as np

from .occupancy import OccupancyResult
from .passability import PassabilityThresholds, classify_passability, occupancy_class

# Trapèze : bas de l'image de 10 % à 90 % de la largeur, sommet à 55 % de la hauteur, de 40 % à 60 %.
CORRIDOR = {"bottom_left": 0.10, "bottom_right": 0.90, "top_y": 0.55, "top_left": 0.40, "top_right": 0.60}
FREE_OBSTACLE_MIN = 0.002   # en dessous de 0,2 % du couloir : considéré sans obstacle (bruit)
TARGET_NAMES = ["occupation", "libre_gauche", "libre_droite"]


def corridor_mask(h: int, w: int, spec: dict = CORRIDOR) -> np.ndarray:
    """Masque booléen (h, w) du couloir de conduite."""
    m = np.zeros((h, w), bool)
    y0 = int(round(spec["top_y"] * (h - 1)))
    for y in range(y0, h):
        t = (y - y0) / max(h - 1 - y0, 1)
        left = (spec["top_left"] + t * (spec["bottom_left"] - spec["top_left"])) * w
        right = (spec["top_right"] + t * (spec["bottom_right"] - spec["top_right"])) * w
        m[y, int(round(left)):int(round(right))] = True
    return m


def corridor_targets(anomaly: np.ndarray, corridor: np.ndarray | None = None, footprint: float = 0.25) -> np.ndarray:
    """[occupation, libre_gauche, libre_droite] (float32) pour un masque d'anomalie (h, w).

    L'espace libre est mesuré à l'EMPRISE AU SOL de l'obstacle : la médiane, sur les 25 % les plus
    bas de ses lignes (la partie la plus proche du véhicule, au contact de la chaussée), de l'écart
    entre le bord du couloir et le premier / dernier pixel d'obstacle de la ligne. Mesurer sur la
    ligne la plus contraignante pénaliserait à tort le haut du couloir, très étroit en perspective."""
    anomaly = anomaly.astype(bool)
    if corridor is None:
        corridor = corridor_mask(*anomaly.shape)
    inside = anomaly & corridor
    occ = float(inside.sum() / corridor.sum())
    if occ < FREE_OBSTACLE_MIN:
        return np.array([occ, 1.0, 1.0], np.float32)
    rows = np.flatnonzero(inside.any(1))
    n_foot = max(1, int(np.ceil(len(rows) * footprint)))
    fl, fr = [], []
    for r in rows[-n_foot:]:
        c = np.flatnonzero(corridor[r]); o = np.flatnonzero(inside[r])
        width = c[-1] - c[0] + 1
        fl.append((o[0] - c[0]) / width)
        fr.append((c[-1] - o[-1]) / width)
    return np.array([occ, float(np.median(fl)), float(np.median(fr))], np.float32)


def classes_from_targets(t, thresholds: PassabilityThresholds = PassabilityThresholds()) -> dict:
    """Classe d'occupation et de passabilité à partir des 3 nombres (vérité terrain ou prédiction)."""
    occ, fl, fr = (float(v) for v in t)
    if occ < FREE_OBSTACLE_MIN:
        return {"occupation": occupancy_class(occ), "passabilite": "voie libre"}
    r = OccupancyResult("ok", occupancy=occ, free_left=fl, free_right=fr,
                        lateral_extent=max(0.0, 1.0 - fl - fr), max_inner_gap=0.0)
    return {"occupation": occupancy_class(occ), "passabilite": classify_passability(r, thresholds)["class"]}


def regression_report(pred: np.ndarray, true: np.ndarray) -> dict:
    """Erreurs absolues moyennes (en points de pourcentage) et accords de classes."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    mae = np.abs(pred - true).mean(0) * 100
    cp = [classes_from_targets(p) for p in pred]
    ct = [classes_from_targets(t) for t in true]
    return {"mae_occupation_pts": float(mae[0]), "mae_libre_gauche_pts": float(mae[1]),
            "mae_libre_droite_pts": float(mae[2]),
            "accord_classe_occupation": float(np.mean([a["occupation"] == b["occupation"] for a, b in zip(cp, ct)])),
            "accord_passabilite": float(np.mean([a["passabilite"] == b["passabilite"] for a, b in zip(cp, ct)])),
            "n": int(len(true))}
