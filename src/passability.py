"""
passability.py — BO3 : classes d'occupation, passabilité qualitative, score de criticité et
message métier.

TOUS LES SEUILS ET POIDS CI-DESSOUS SONT DES HYPOTHÈSES DE TRAVAIL (valeurs par défaut),
pas des résultats. Ils doivent être discutés, ajustés sur la validation (ou avec un expert
métier) et présentés comme tels dans le rapport. Sans calibration caméra, la passabilité est
une classe qualitative dérivée de rapports de pixels, jamais une distance en mètres.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .occupancy import OccupancyResult

FAMILY_LABELS = {"faune": "faune (animal)", "debris_objets_perdus": "débris / objet perdu",
                 "infrastructure_temporaire": "infrastructure temporaire", "autre": "autre",
                 None: "non déterminé"}

# Seuils d'occupation demandés par le cahier des charges : <5 %, 5–15 %, 15–30 %, >30 %
OCCUPANCY_BINS = [(0.05, "faible"), (0.15, "modérée"), (0.30, "forte"), (float("inf"), "critique")]


@dataclass
class PassabilityThresholds:
    side_ok: float = 0.30       # espace libre d'un côté (fraction de largeur de chaussée) pour passer
    side_comfort: float = 0.45  # au-delà : « marge suffisante », sinon « marge réduite »
    side_tight: float = 0.15    # en dessous partout : voie bloquée
    blocked_lateral: float = 0.90   # emprise latérale au-delà de laquelle la voie est bloquée
    critical_occupancy: float = 0.30  # occupation globale au-delà : au mieux « passage difficile »

    def to_dict(self) -> dict:
        return asdict(self)


def occupancy_class(occ: float | None) -> str:
    if occ is None:
        return "indéterminée"
    for upper, label in OCCUPANCY_BINS:
        if occ < upper:
            return label
    return OCCUPANCY_BINS[-1][1]


def classify_passability(r: OccupancyResult, t: PassabilityThresholds = PassabilityThresholds()) -> dict:
    """Renvoie {'class', 'side', 'margin', 'reason'} parmi :
    'voie libre', 'passage des deux côtés', 'passage à gauche', 'passage à droite',
    'passage difficile', 'voie bloquée', 'indéterminé'."""
    if r.status == "no_road":
        return {"class": "indéterminé", "side": None, "margin": None,
                "reason": "chaussée non détectée : passabilité non calculable"}
    if r.status == "no_obstacle_on_road" or r.free_left is None:
        return {"class": "voie libre", "side": None, "margin": None,
                "reason": "aucune anomalie sur la chaussée reconstruite"}
    left_ok, right_ok = r.free_left >= t.side_ok, r.free_right >= t.side_ok
    best_free = max(r.free_left, r.free_right, r.max_inner_gap or 0.0)

    if r.lateral_extent >= t.blocked_lateral or best_free < t.side_tight:
        cls, side = "voie bloquée", None
    elif left_ok and right_ok:
        cls, side = "passage des deux côtés", "left" if r.free_left >= r.free_right else "right"
    elif left_ok:
        cls, side = "passage à gauche", "left"
    elif right_ok:
        cls, side = "passage à droite", "right"
    else:
        cls, side = "passage difficile", None

    reason = f"libre à gauche {r.free_left:.0%}, à droite {r.free_right:.0%} de la largeur"
    if (r.occupancy or 0) > t.critical_occupancy and cls in ("passage des deux côtés", "passage à gauche",
                                                              "passage à droite"):
        cls, reason = "passage difficile", reason + " ; occupation globale critique"
    margin = None
    if side is not None and cls != "passage difficile":
        free = r.free_left if side == "left" else r.free_right
        margin = "suffisante" if free >= t.side_comfort else "réduite"
    return {"class": cls, "side": side, "margin": margin, "reason": reason}


# Pondérations HEURISTIQUES du score de criticité (somme = 1) — à justifier dans le rapport.
CRITICALITY_WEIGHTS = {"occupancy": 0.35, "passability": 0.35, "proximity": 0.20, "family": 0.10}
PASSABILITY_RISK = {"voie libre": 0.0, "passage des deux côtés": 0.1, "passage à gauche": 0.4,
                    "passage à droite": 0.4, "passage difficile": 0.75, "voie bloquée": 1.0, "indéterminé": 0.5}
# Faune : mobile et imprévisible ; débris : statique, rarement signalé ; infrastructure : signalée.
FAMILY_RISK = {"faune": 1.0, "debris_objets_perdus": 0.8, "infrastructure_temporaire": 0.5, "autre": 0.7, None: 0.7}
CRITICALITY_LEVELS = [(25, "faible"), (50, "modérée"), (75, "élevée"), (101, "critique")]


def criticality_score(r: OccupancyResult, passability: dict, family: str | None = None,
                      t: PassabilityThresholds = PassabilityThresholds()) -> dict:
    """Score 0–100 = somme pondérée de composantes normalisées dans [0, 1]."""
    comp = {
        "occupancy": min((r.occupancy or 0.0) / t.critical_occupancy, 1.0),
        "passability": PASSABILITY_RISK[passability["class"]],
        "proximity": r.proximity if r.proximity is not None else 0.5,
        "family": FAMILY_RISK.get(family, FAMILY_RISK[None]),
    }
    if passability["class"] == "voie libre":
        comp["proximity"] = comp["family"] = 0.0
    score = 100 * sum(CRITICALITY_WEIGHTS[k] * v for k, v in comp.items())
    level = next(lbl for upper, lbl in CRITICALITY_LEVELS if score < upper)
    return {"score": round(float(score), 1), "level": level, "components": comp}


def business_message(r: OccupancyResult, passability: dict, family: str | None = None,
                     traffic_side: str = "right") -> str:
    """Message métier construit UNIQUEMENT à partir des valeurs calculées."""
    if passability["class"] == "indéterminé":
        return "Chaussée non détectée : impossible d'évaluer l'emprise et la passabilité (vérification manuelle)."
    if passability["class"] == "voie libre":
        return "Aucune anomalie détectée sur la chaussée visible."
    occ = f"{100 * r.occupancy:.1f} %"
    head = (f"Obstacle de type {FAMILY_LABELS.get(family, family)}" if family else "Obstacle détecté")
    cls, margin = passability["class"], passability["margin"]
    if cls == "voie bloquée":
        advice = "voie bloquée : arrêt recommandé"
    elif cls == "passage difficile":
        advice = "passage difficile : ralentir fortement et contourner avec prudence"
    elif cls == "passage des deux côtés":
        advice = f"passage possible des deux côtés (côté le plus large : {'gauche' if passability['side'] == 'left' else 'droite'}, marge {margin})"
    else:
        side = "gauche" if passability["side"] == "left" else "droite"
        advice = f"passage recommandé à {side} avec marge {margin}"
        if (traffic_side == "right" and side == "gauche") or (traffic_side == "left" and side == "droite"):
            advice += " (empiète sur la voie opposée : vérifier le trafic venant en face)"
    return f"{head}, occupe {occ} de la chaussée visible, {advice}."


def analyse_scene(road, anomaly, family: str | None = None,
                  thresholds: PassabilityThresholds = PassabilityThresholds(), traffic_side: str = "right") -> dict:
    """Raccourci : occupation + classe d'occupation + passabilité + criticité + message."""
    from .occupancy import compute_occupancy
    r, region, on_road = compute_occupancy(road, anomaly)
    p = classify_passability(r, thresholds)
    out = {"occupancy": r.to_dict(), "occupancy_class": occupancy_class(r.occupancy), "passability": p,
           "criticality": criticality_score(r, p, family, thresholds),
           "message": business_message(r, p, family, traffic_side), "thresholds": thresholds.to_dict()}
    return out | {"_region": region, "_on_road": on_road}
