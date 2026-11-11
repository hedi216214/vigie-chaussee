"""
occupancy.py — BO3 : emprise de l'anomalie sur la chaussée visible (sans calibration caméra).

Entrées : masque de chaussée (classe « road » de DeepLabV3+ MobileNetV2 Cityscapes) et masque
d'anomalie (meilleur modèle BO2), à la même résolution.

RoadAnomaly ne fournit NI masque de chaussée NI calibration caméra. Conséquences assumées :
    * toutes les grandeurs sont des RAPPORTS DE PIXELS dans l'image (pas des m² ni des mètres) ;
    * la perspective écrase les objets lointains : un même objet occupe un % plus faible
      quand il est loin. `occupancy_local` (calculé sur les seules lignes de l'obstacle) et
      `proximity` (position verticale) servent à nuancer ce biais ;
    * la chaussée cachée par l'obstacle est reconstruite par interpolation des bords de route
      des lignes non occultées (hypothèse : bords localement réguliers ; fragile en virage,
      intersection ou route non goudronnée mal segmentée — domain gap Cityscapes).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage

ROAD_ID, SIDEWALK_ID = 0, 1   # trainIds Cityscapes


def road_mask_from_cityscapes(pred_ids: np.ndarray, include_sidewalk: bool = False) -> np.ndarray:
    """Carte de classes Cityscapes (H, W) -> masque booléen de chaussée."""
    road = pred_ids == ROAD_ID
    return road | (pred_ids == SIDEWALK_ID) if include_sidewalk else road


def main_road_component(road: np.ndarray, bottom_frac: float = 0.2) -> np.ndarray:
    """Plus grande composante de route touchant le bas de l'image (route de l'égo-véhicule) ;
    à défaut, la plus grande composante."""
    lab, k = ndimage.label(road)
    if k == 0:
        return np.zeros_like(road, dtype=bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    h = road.shape[0]
    bottom_ids = np.unique(lab[int(h * (1 - bottom_frac)):])
    bottom_ids = bottom_ids[bottom_ids > 0]
    best = bottom_ids[np.argmax(sizes[bottom_ids])] if len(bottom_ids) else int(np.argmax(sizes))
    return lab == best


def anomalies_on_road(anomaly: np.ndarray, road: np.ndarray, touch_px: int | None = None) -> np.ndarray:
    """Garde les composantes d'anomalie qui touchent la chaussée (dilatée de touch_px).
    Un animal sur le bas-côté, sans contact avec la route, est exclu."""
    if touch_px is None:
        touch_px = max(2, road.shape[1] // 100)
    near = ndimage.binary_dilation(road, iterations=touch_px)
    lab, k = ndimage.label(anomaly)
    if k == 0:
        return np.zeros_like(anomaly, dtype=bool)
    keep = np.unique(lab[near & (lab > 0)])
    return np.isin(lab, keep[keep > 0])


def _boundaries(mask: np.ndarray, rows: np.ndarray):
    left, right = [], []
    for r in rows:
        cols = np.flatnonzero(mask[r])
        left.append(cols[0])
        right.append(cols[-1])
    return np.asarray(left, float), np.asarray(right, float)


def _fill_edges(rows_all, rows_ok, vals_ok, k_fit: int = 30):
    """Interpolation à l'intérieur, extrapolation linéaire (régression sur les k lignes les plus
    proches) à l'extérieur de l'intervalle des lignes non occultées."""
    out = np.interp(rows_all, rows_ok, vals_ok)
    for side, sel in (("low", rows_all < rows_ok[0]), ("high", rows_all > rows_ok[-1])):
        if sel.any() and len(rows_ok) >= 2:
            idx = slice(0, k_fit) if side == "low" else slice(-k_fit, None)
            a, b = np.polyfit(rows_ok[idx], vals_ok[idx], 1)
            out[sel] = a * rows_all[sel] + b
    return out


def drivable_region(road: np.ndarray, anomaly_on_road: np.ndarray, min_row_px: int = 3):
    """Chaussée visible reconstruite (H, W) : pour chaque ligne entre le haut et le bas de la
    route, intervalle [bord gauche, bord droit] ; bords des lignes occultées par l'obstacle
    interpolés depuis les lignes libres. Renvoie (région, bords gauche, bords droit, lignes)."""
    h, w = road.shape
    road_rows = np.flatnonzero(road.sum(1) >= min_row_px)
    if len(road_rows) == 0:
        return np.zeros_like(road, dtype=bool), None, None, None
    obst_rows = np.flatnonzero(anomaly_on_road.any(1))
    top = min(road_rows[0], obst_rows[0]) if len(obst_rows) else road_rows[0]
    bottom = max(road_rows[-1], obst_rows[-1]) if len(obst_rows) else road_rows[-1]
    rows_all = np.arange(top, bottom + 1)

    blocked = np.zeros(h, bool)
    if len(obst_rows):
        blocked[max(0, obst_rows[0] - 2): obst_rows[-1] + 3] = True   # petite marge verticale
    clean = road_rows[~blocked[road_rows]]
    if len(clean) >= 2:
        l_ok, r_ok = _boundaries(road, clean)
        left = _fill_edges(rows_all, clean.astype(float), l_ok)
        right = _fill_edges(rows_all, clean.astype(float), r_ok)
    else:  # repli : enveloppe ligne par ligne de route ∪ obstacle
        union = road | anomaly_on_road
        rows_u = rows_all[union[rows_all].any(1)]
        lu, ru = _boundaries(union, rows_u)
        left, right = np.interp(rows_all, rows_u, lu), np.interp(rows_all, rows_u, ru)
    left = np.clip(np.round(left), 0, w - 1).astype(int)
    right = np.clip(np.round(right), 0, w - 1).astype(int)
    right = np.maximum(right, left)

    region = np.zeros_like(road, dtype=bool)
    for r, a, b in zip(rows_all, left, right):
        region[r, a:b + 1] = True
    region |= road            # la route réellement segmentée fait toujours partie de la chaussée
    return region, left, right, rows_all


@dataclass
class OccupancyResult:
    status: str                       # "ok" | "no_road" | "no_obstacle_on_road"
    road_px: int = 0
    anomaly_on_road_px: int = 0
    n_obstacles_on_road: int = 0
    n_anomalies_off_road: int = 0
    occupancy: float | None = None          # surface anomalie / surface de chaussée visible
    occupancy_local: float | None = None    # idem, restreint aux lignes de l'obstacle
    lateral_extent: float | None = None     # largeur obstacle / largeur chaussée (p90 sur les lignes)
    longitudinal_extent: float | None = None  # hauteur obstacle / hauteur de chaussée visible
    free_left: float | None = None          # espace libre à gauche / largeur chaussée (p10)
    free_right: float | None = None         # espace libre à droite / largeur chaussée (p10)
    max_inner_gap: float | None = None      # plus grand passage entre deux morceaux d'obstacle (p10)
    proximity: float | None = None          # 0 = loin (haut de la chaussée), 1 = près (bas de l'image)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _largest_gap(row_obst: np.ndarray, a: int, b: int) -> int:
    """Plus long segment libre strictement entre le premier et le dernier pixel d'obstacle."""
    cols = np.flatnonzero(row_obst[a:b + 1])
    if len(cols) < 2:
        return 0
    return int(np.max(np.diff(cols)) - 1)


def compute_occupancy(road: np.ndarray, anomaly: np.ndarray, min_road_frac: float = 0.01,
                      low_q: float = 10, high_q: float = 90) -> tuple[OccupancyResult, np.ndarray, np.ndarray]:
    """Pipeline BO3 complet. Renvoie (résultat, région de chaussée, anomalie sur chaussée).
    Les quantiles (p10 / p90 sur les lignes de l'obstacle) rendent les mesures robustes au bruit
    de segmentation tout en restant proches de la ligne la plus contraignante."""
    road = main_road_component(road.astype(bool))
    anomaly = anomaly.astype(bool)
    h, w = road.shape
    if road.mean() < min_road_frac:
        return OccupancyResult("no_road", notes=["chaussée non détectée ou trop petite"]), road, np.zeros_like(road)

    on_road = anomalies_on_road(anomaly, road)
    _, k_all = ndimage.label(anomaly)
    _, k_on = ndimage.label(on_road)
    region, left, right, rows_all = drivable_region(road, on_road)
    on_road &= region
    res = OccupancyResult("ok", road_px=int(region.sum()), anomaly_on_road_px=int(on_road.sum()),
                          n_obstacles_on_road=int(k_on), n_anomalies_off_road=int(k_all - k_on))
    if on_road.sum() == 0:
        res.status, res.occupancy = "no_obstacle_on_road", 0.0
        return res, region, on_road

    res.occupancy = float(on_road.sum() / region.sum())
    obst_rows = np.flatnonzero(on_road.any(1))
    res.occupancy_local = float(on_road[obst_rows].sum() / max(region[obst_rows].sum(), 1))

    idx = {r: i for i, r in enumerate(rows_all)}
    fl, fr, ext, gap = [], [], [], []
    for r in obst_rows:
        i = idx.get(r)
        if i is None:
            continue
        a, b = left[i], right[i]
        width = b - a + 1
        cols = np.flatnonzero(on_road[r, a:b + 1])
        if len(cols) == 0 or width <= 1:
            continue
        o_l, o_r = a + cols[0], a + cols[-1]
        fl.append((o_l - a) / width)
        fr.append((b - o_r) / width)
        ext.append((o_r - o_l + 1) / width)
        gap.append(_largest_gap(on_road[r], o_l, o_r) / width)
    if not fl:
        res.notes.append("obstacle hors de l'intervalle de chaussée reconstruit")
        return res, region, on_road

    res.free_left = float(np.percentile(fl, low_q))
    res.free_right = float(np.percentile(fr, low_q))
    res.lateral_extent = float(np.percentile(ext, high_q))
    res.max_inner_gap = float(np.percentile(gap, low_q))
    span = rows_all[-1] - rows_all[0] + 1
    res.longitudinal_extent = float((obst_rows[-1] - obst_rows[0] + 1) / span)
    res.proximity = float((obst_rows[-1] - rows_all[0]) / max(span - 1, 1))
    if res.n_anomalies_off_road:
        res.notes.append(f"{res.n_anomalies_off_road} anomalie(s) hors chaussée ignorée(s)")
    return res, region, on_road
