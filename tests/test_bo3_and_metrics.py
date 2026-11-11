"""
Tests unitaires sur des masques SYNTHÉTIQUES (géométries construites à la main).
Ils vérifient la logique du code (BO3, métriques, splits) ; ce ne sont pas des résultats
expérimentaux sur RoadAnomaly.

    python -m pytest tests -q
Variable d'environnement optionnelle : ROADANOMALY_ROOT=<chemin vers RoadAnomaly/> pour
activer les tests de lecture du vrai dataset (sinon ils sont ignorés).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluate import binary_classification_metrics, segmentation_metrics  # noqa: E402
from src.occupancy import compute_occupancy  # noqa: E402
from src.passability import analyse_scene, occupancy_class  # noqa: E402

H, W = 288, 512


def trapezoid_road(top=120, top_half=60, bottom_half=240, cx=256):
    """Route en perspective : trapèze du haut (étroit) vers le bas de l'image (large)."""
    road = np.zeros((H, W), bool)
    for r in range(top, H):
        t = (r - top) / (H - 1 - top)
        half = top_half + t * (bottom_half - top_half)
        road[r, int(max(0, cx - half)):int(min(W, cx + half))] = True
    return road


def box(r0, r1, c0, c1):
    m = np.zeros((H, W), bool)
    m[r0:r1, c0:c1] = True
    return m


def test_obstacle_on_right_gives_pass_left():
    road = trapezoid_road()
    obst = box(200, 240, 300, 420) & road
    out = analyse_scene(road & ~obst, obst)          # la route est cachée sous l'obstacle
    occ = out["occupancy"]
    assert occ["status"] == "ok"
    assert occ["free_left"] > occ["free_right"]
    assert out["passability"]["class"] == "passage à gauche"
    assert "gauche" in out["message"] and "%" in out["message"]


def test_small_central_obstacle_two_sides():
    road = trapezoid_road()
    obst = box(230, 245, 245, 268)
    out = analyse_scene(road & ~obst, obst)
    assert out["passability"]["class"] == "passage des deux côtés"
    assert out["occupancy_class"] == "faible"


def test_full_width_obstacle_blocks():
    road = trapezoid_road()
    obst = box(200, 230, 0, W) & trapezoid_road(top=0, top_half=400, bottom_half=400)
    out = analyse_scene(road & ~obst, obst)
    assert out["passability"]["class"] == "voie bloquée"
    assert out["criticality"]["score"] > 50


def test_occluded_road_is_reconstructed():
    """L'occupation doit être calculée par rapport à la chaussée SOUS l'obstacle aussi."""
    road = trapezoid_road()
    obst = box(200, 240, 200, 300)
    r, region, on_road = compute_occupancy(road & ~obst, obst)
    assert region[220, 250]                          # pixel caché par l'obstacle -> chaussée
    expected = obst.sum() / road.sum()
    assert abs(r.occupancy - expected) / expected < 0.15


def test_no_road_is_undetermined():
    out = analyse_scene(np.zeros((H, W), bool), box(100, 150, 100, 150))
    assert out["passability"]["class"] == "indéterminé"
    assert "impossible" in out["message"]


def test_off_road_anomaly_is_ignored():
    road = trapezoid_road(top_half=30, bottom_half=80)
    off = box(130, 160, 0, 40)                       # bord gauche, loin de la route
    out = analyse_scene(road, off)
    assert out["passability"]["class"] == "voie libre"
    assert out["occupancy"]["n_anomalies_off_road"] == 1


def test_occupancy_bins_match_spec():
    assert [occupancy_class(v) for v in (0.01, 0.05, 0.10, 0.20, 0.31)] == \
        ["faible", "modérée", "modérée", "forte", "critique"]


def test_segmentation_metrics_perfect_and_disjoint():
    gt = box(10, 20, 10, 20).astype(np.uint8)[None]
    assert segmentation_metrics(gt.astype(float), gt, 0.5)["iou"] == pytest.approx(1.0)
    other = box(50, 60, 50, 60).astype(float)[None]
    assert segmentation_metrics(other, gt, 0.5)["iou"] == pytest.approx(0.0, abs=1e-6)


def test_classification_metrics_shapes():
    y = np.array([0, 0, 1, 1])
    m = binary_classification_metrics(y, np.array([0.1, 0.6, 0.4, 0.9]), 0.5)
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]
    assert 0 <= m["auroc"] <= 1


ROOT = os.environ.get("ROADANOMALY_ROOT")


@pytest.mark.skipif(not ROOT, reason="ROADANOMALY_ROOT non défini")
def test_real_dataset_reading_and_splits():
    from src.dataset import audit_annotations, check_no_leakage, list_frames, load_mask, make_splits
    names = list_frames(ROOT)
    assert len(names) > 0
    m = load_mask(ROOT, names[0], (W, H))
    assert m.shape == (H, W) and set(np.unique(m)) <= {0, 1}
    splits = make_splits(names)
    check_no_leakage(splits)
    assert sorted(sum(splits.values(), [])) == sorted(names)
    df = audit_annotations(ROOT, names[:5])
    assert df["error"].isna().all()


def test_webapp_pipeline_with_fake_predictors():
    """Ancien démonstrateur (archive/) : teste le câblage de l'application avec des prédicteurs factices (aucun modèle réel)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "archive", "webapp_gradio_ancienne_version"))
    from pipeline import analyse_image
    img = np.full((720, 1280, 3), 120, np.uint8)
    road = trapezoid_road()
    obst = box(200, 240, 300, 420) & road
    ids = np.where(road & ~obst, 0, 10).astype(np.uint8)        # 0 = road, 10 = sky (Cityscapes)

    vis, heat, md, rep = analyse_image(img, (W, H), lambda im: obst.astype(np.float32), 0.5,
                                       road_fn=lambda im: ids, family_fn=lambda crop: ("faune", 0.9))
    assert vis.shape == (H, W, 3) and heat.shape == (H, W, 3)
    assert rep["famille"]["valeur"] == "faune"
    assert rep["passability"]["class"] == "passage à gauche"
    assert "faune" in md
    _, _, md2, rep2 = analyse_image(img, (W, H), lambda im: obst.astype(np.float32), 0.5)
    assert "indisponible" in md2 and "passability" not in rep2
