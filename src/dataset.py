"""
dataset.py — Chargement, audit, taxonomie, découpage et Datasets PyTorch pour RoadAnomaly.

Ce que dit la documentation officielle (readme.md du dataset, Lis et al., ICCV 2019) :
    RoadAnomaly/
        frame_list.json                 liste des noms d'images
        credits.txt                     sources des images
        frames/<nom>.webp               image
        frames/<nom>.labels/labels_semantic.png   fond = 0, anomalie = 2
        frames/<nom>.labels/labels_instance.png   identifiant d'instance (absent pour certaines images)

Tout le reste (nombre d'images, résolution, valeurs réellement présentes dans les masques,
taille des objets…) doit être VÉRIFIÉ sur votre copie locale avec `audit_annotations()`.

La famille métier de chaque image (faune / débris / infrastructure temporaire / autre) n'est
PAS une annotation du dataset : elle est déduite du préfixe du nom de fichier par une règle
explicite (FAMILY_RULES) et doit être contrôlée visuellement (notebook 01).
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
import zipfile
from typing import Iterable

import numpy as np
from PIL import Image

try:  # le module reste utilisable sans PyTorch (exploration, tests)
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object

from .config import DATASET_URL, IMAGENET_MEAN, IMAGENET_STD

ANOMALY_LABEL = 2
BACKGROUND_LABEL = 0

# --------------------------------------------------------------------------------------
# 1. Taxonomie métier (BO1)
# --------------------------------------------------------------------------------------
FAMILIES = ["faune", "debris_objets_perdus", "infrastructure_temporaire", "autre"]

# Règle : préfixe du nom de fichier -> famille métier. Hypothèse à vérifier visuellement.
FAMILY_RULES = [
    (r"^animals", "faune"),
    (r"^cones", "infrastructure_temporaire"),
    (r"czech_hedgehog", "infrastructure_temporaire"),   # barrière anti-véhicule : choix discutable
    (r"^lost", "debris_objets_perdus"),
    (r"^obstacles", "debris_objets_perdus"),             # rochers, blocs, objets sur la voie
    (r"^vehicle", "autre"),                              # véhicules atypiques : hors des 3 familles
]
# Corrections manuelles après inspection visuelle : {"nom_image": "famille"}
FAMILY_OVERRIDES: dict[str, str] = {}

# Images d'une même scène / vidéo : elles doivent rester dans le même sous-ensemble.
GROUP_RULES = [r"obstacles12_rocks", r"lost_tires", r"cardboard_czech_hedgehog"]


def family_of(name: str) -> str:
    if name in FAMILY_OVERRIDES:
        return FAMILY_OVERRIDES[name]
    for pattern, family in FAMILY_RULES:
        if re.search(pattern, name):
            return family
    return "autre"


def category_of(name: str) -> str:
    """Catégorie du nom de fichier (animals, cones, lost, obstacles, vehicle)."""
    return re.match(r"[a-z]+", name).group(0)


def group_of(name: str) -> str:
    for pattern in GROUP_RULES:
        if re.search(pattern, name):
            return pattern
    return name


# --------------------------------------------------------------------------------------
# 2. Téléchargement et lecture
# --------------------------------------------------------------------------------------
def ensure_dataset(data_dir: str, download: bool = True) -> str:
    """Renvoie le chemin de RoadAnomaly/. Décompresse ou télécharge si nécessaire."""
    root = os.path.join(data_dir, "RoadAnomaly")
    if os.path.isfile(os.path.join(root, "frame_list.json")):
        return root
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "RoadAnomaly.zip")
    if not os.path.isfile(zip_path):
        if not download:
            raise FileNotFoundError(f"Dataset introuvable dans {data_dir} (ni dossier, ni zip).")
        print(f"Téléchargement de {DATASET_URL} …")
        urllib.request.urlretrieve(DATASET_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    if not os.path.isfile(os.path.join(root, "frame_list.json")):
        raise FileNotFoundError("Archive décompressée mais structure inattendue : vérifiez le contenu du zip.")
    return root


def list_frames(root: str) -> list[str]:
    """Noms des images (sans extension), dans l'ordre de frame_list.json."""
    with open(os.path.join(root, "frame_list.json")) as f:
        return [os.path.splitext(x)[0] for x in json.load(f)]


def image_path(root: str, name: str) -> str:
    for ext in (".webp", ".jpg", ".png"):
        p = os.path.join(root, "frames", name + ext)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Image introuvable pour {name}")


def mask_path(root: str, name: str, kind: str = "semantic") -> str:
    return os.path.join(root, "frames", f"{name}.labels", f"labels_{kind}.png")


def load_image(root: str, name: str, size: tuple | None = None) -> np.ndarray:
    """Image RGB uint8 (H, W, 3) ; size = (largeur, hauteur), interpolation bilinéaire."""
    img = Image.open(image_path(root, name)).convert("RGB")
    if size is not None and img.size != tuple(size):
        img = img.resize(tuple(size), Image.BILINEAR)
    return np.asarray(img)


def load_mask(root: str, name: str, size: tuple | None = None) -> np.ndarray:
    """Masque binaire uint8 (1 = anomalie). Redimensionnement au plus proche voisin
    (une interpolation bilinéaire créerait des valeurs de classe inexistantes)."""
    m = Image.open(mask_path(root, name))
    if size is not None and m.size != tuple(size):
        m = m.resize(tuple(size), Image.NEAREST)
    return (np.asarray(m) == ANOMALY_LABEL).astype(np.uint8)


def load_arrays(root: str, names: Iterable[str], size: tuple) -> tuple[np.ndarray, np.ndarray]:
    names = list(names)
    images = np.stack([load_image(root, n, size) for n in names])
    masks = np.stack([load_mask(root, n, size) for n in names])
    return images, masks


# --------------------------------------------------------------------------------------
# 3. Audit de la qualité des annotations (BO1 / DSO « audit »)
# --------------------------------------------------------------------------------------
def audit_annotations(root: str, names: list[str], tiny_px: int = 4):
    """Contrôles automatiques, une ligne par image (DataFrame pandas).

    Colonnes : taille image/masque, cohérence, valeurs présentes dans le masque, fraction
    de pixels anormaux, nombre de composantes connexes, composantes minuscules (< tiny_px,
    candidates au bruit d'annotation), présence du masque d'instances, famille déduite.
    Aucun chiffre n'est supposé : tout est calculé sur votre copie du dataset.
    """
    import pandas as pd
    from scipy import ndimage

    rows = []
    for n in names:
        row = {"name": n, "family": family_of(n), "group": group_of(n)}
        try:
            img = Image.open(image_path(root, n))
            sem = np.asarray(Image.open(mask_path(root, n)))
            row.update(img_w=img.size[0], img_h=img.size[1], mask_w=sem.shape[1], mask_h=sem.shape[0])
            row["size_match"] = (img.size[0], img.size[1]) == (sem.shape[1], sem.shape[0])
            values = np.unique(sem)
            row["mask_values"] = ",".join(map(str, values.tolist()))
            row["unexpected_values"] = bool(set(values.tolist()) - {BACKGROUND_LABEL, ANOMALY_LABEL})
            binary = sem == ANOMALY_LABEL
            row["anomaly_fraction"] = float(binary.mean())
            lab, k = ndimage.label(binary)
            sizes = np.bincount(lab.ravel())[1:] if k else np.array([])
            row["n_components"] = int(k)
            row["n_tiny_components"] = int((sizes < tiny_px).sum())
            row["largest_component_px"] = int(sizes.max()) if k else 0
            row["has_instance_mask"] = os.path.isfile(mask_path(root, n, "instance"))
            row["error"] = None
        except Exception as e:  # on signale l'erreur au lieu d'interrompre l'audit
            row["error"] = repr(e)
        rows.append(row)
    return pd.DataFrame(rows)


def object_sizes(masks: np.ndarray, names: list[str]):
    """Une ligne par composante connexe (objet) : aire en pixels à la résolution des masques."""
    import pandas as pd
    from scipy import ndimage

    rows = []
    for m, n in zip(masks, names):
        lab, k = ndimage.label(m > 0)
        for i, area in enumerate(np.bincount(lab.ravel())[1:], start=1):
            rows.append({"name": n, "family": family_of(n), "component": i, "area_px": int(area)})
    return pd.DataFrame(rows)


def size_bin(area_px: float, small: int = 32 ** 2, large: int = 96 ** 2) -> str:
    """Classes de taille de type COCO (seuils exprimés à la résolution de travail)."""
    return "petit" if area_px < small else ("moyen" if area_px < large else "grand")


# --------------------------------------------------------------------------------------
# 4. Découpage train / val / test (stratifié par famille, groupé par scène)
# --------------------------------------------------------------------------------------
def make_splits(names: list[str], n_val: int = 10, n_test: int = 10, seed: int = 42,
                stratify: str = "category") -> dict:
    """Les scènes multi-images partent en train (pas de fuite de scène vers val/test) ;
    les images isolées sont réparties dans val/test proportionnellement à chaque famille
    (méthode des plus forts restes). Le split est sauvegardé par le notebook 02 et réutilisé
    tel quel par tous les autres notebooks."""
    rng = np.random.default_rng(seed)
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(group_of(n), []).append(n)
    train = [n for g in groups.values() if len(g) > 1 for n in g]
    singles = [g[0] for g in groups.values() if len(g) == 1]

    key = category_of if stratify == "category" else family_of
    by_fam: dict[str, list[str]] = {}
    for n in singles:
        by_fam.setdefault(key(n), []).append(n)
    fams = sorted(by_fam)

    def allocate(total: int) -> dict:
        sizes = np.array([len(by_fam[f]) for f in fams], dtype=float)
        exact = sizes / sizes.sum() * total
        alloc = np.floor(exact).astype(int)
        for i in np.argsort(-(exact - alloc))[: total - alloc.sum()]:
            alloc[i] += 1
        return dict(zip(fams, alloc))

    a_test, a_val = allocate(n_test), allocate(n_val)
    val, test = [], []
    for f in fams:
        items = list(by_fam[f])
        rng.shuffle(items)
        test += items[: a_test[f]]
        val += items[a_test[f]: a_test[f] + a_val[f]]
        train += items[a_test[f] + a_val[f]:]
    order = {n: i for i, n in enumerate(names)}
    splits = {k: sorted(v, key=order.get) for k, v in {"train": train, "val": val, "test": test}.items()}
    check_no_leakage(splits)
    return splits


def check_no_leakage(splits: dict) -> None:
    """Lève une erreur si une image ou une scène apparaît dans deux sous-ensembles."""
    seen: dict[str, str] = {}
    for split, items in splits.items():
        for n in items:
            for key in (n, "group:" + group_of(n)):
                if key in seen and seen[key] != split:
                    raise ValueError(f"Fuite de données : {key} dans {seen[key]} et {split}")
                seen[key] = split


# --------------------------------------------------------------------------------------
# 5. Prétraitement et augmentation
# --------------------------------------------------------------------------------------
def normalize(image_uint8: np.ndarray, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """uint8 (H,W,3) -> float32 (3,H,W) normalisé ImageNet (convention des modèles pré-entraînés)."""
    x = image_uint8.astype(np.float32) / 255.0
    x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return x.transpose(2, 0, 1).copy()


def augment_pair(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                 p_flip: float = 0.5, scale_range=(0.6, 1.0), color: float = 0.25):
    """Même transformation géométrique pour l'image et le masque.
    Flip horizontal (pas vertical : la route reste en bas), zoom aléatoire (distance à
    l'obstacle), variations photométriques sur l'image seulement."""
    H, W = mask.shape
    if rng.random() < p_flip:
        image, mask = image[:, ::-1], mask[:, ::-1]
    s = rng.uniform(*scale_range)
    ch, cw = int(H * s), int(W * s)
    y0, x0 = rng.integers(0, H - ch + 1), rng.integers(0, W - cw + 1)
    image, mask = image[y0:y0 + ch, x0:x0 + cw], mask[y0:y0 + ch, x0:x0 + cw]
    if (ch, cw) != (H, W):
        image = np.asarray(Image.fromarray(np.ascontiguousarray(image)).resize((W, H), Image.BILINEAR))
        mask = np.asarray(Image.fromarray(np.ascontiguousarray(mask)).resize((W, H), Image.NEAREST))
    if color > 0:
        img = image.astype(np.float32) / 255.0
        img = img * (1 + rng.uniform(-color, color))
        img = (img - img.mean()) * (1 + rng.uniform(-color, color)) + img.mean()
        gray = img.mean(axis=2, keepdims=True)
        img = (img - gray) * (1 + rng.uniform(-color, color)) + gray
        image = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class SegmentationDataset(Dataset):
    """Paires (image normalisée, masque binaire) pour la segmentation (BO2).
    `repeats` répète le jeu d'entraînement par époque (petit dataset + augmentation)."""

    def __init__(self, images: np.ndarray, masks: np.ndarray, augment: bool = False,
                 repeats: int = 1, seed: int = 42):
        self.images, self.masks, self.augment, self.repeats = images, masks, augment, repeats
        self.seed = seed

    def __len__(self) -> int:
        return len(self.images) * self.repeats

    def __getitem__(self, i: int):
        j = i % len(self.images)
        img, m = self.images[j], self.masks[j]
        if self.augment:  # graine dérivée de torch -> reproductible avec torch.manual_seed
            rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
            img, m = augment_pair(img, m, rng)
        return torch.from_numpy(normalize(img)), torch.from_numpy(m.astype(np.float32))[None]


# --------------------------------------------------------------------------------------
# 6. Patches pour la classification présence / absence (BO1)
# --------------------------------------------------------------------------------------
def extract_patches(images: np.ndarray, masks: np.ndarray, names: list[str], patch: int = 96,
                    stride: int = 48, pos_min: float = 0.05, neg_max: float = 0.0):
    """Découpe chaque image en patches et les étiquette :
        1 (anomalie présente) si la fraction de pixels anormaux >= pos_min,
        0 (absente)          si elle est <= neg_max,
        ignoré sinon (zone ambiguë, bord d'objet).

    Pourquoi : TOUTES les images de RoadAnomaly contiennent une anomalie ; une
    classification présence/absence au niveau image n'aurait donc aucun exemple négatif.
    Les patches doivent être extraits APRÈS le split (patches d'une même image = même split).
    """
    X, y, meta = [], [], []
    H, W = masks.shape[1:]
    for img, m, n in zip(images, masks, names):
        for y0 in range(0, H - patch + 1, stride):
            for x0 in range(0, W - patch + 1, stride):
                frac = float(m[y0:y0 + patch, x0:x0 + patch].mean())
                if frac >= pos_min:
                    label = 1
                elif frac <= neg_max:
                    label = 0
                else:
                    continue
                X.append(img[y0:y0 + patch, x0:x0 + patch])
                y.append(label)
                meta.append({"name": n, "family": family_of(n), "x0": x0, "y0": y0, "frac": frac})
    return np.stack(X), np.asarray(y, np.int64), meta


def largest_component_crop(image: np.ndarray, mask: np.ndarray, margin: float = 0.25):
    """Recadrage (boîte englobante + marge relative) de la plus grande composante du masque ;
    None si le masque est vide. Sert au classifieur de famille (BO1-b) et à l'application."""
    from scipy import ndimage
    lab, k = ndimage.label(mask)
    if k == 0:
        return None
    biggest = 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))
    rows, cols = np.where(lab == biggest)
    h, w = mask.shape
    mr = max(4, int((np.ptp(rows) + 1) * margin))
    mc = max(4, int((np.ptp(cols) + 1) * margin))
    return image[max(0, rows.min() - mr):min(h, rows.max() + mr + 1),
                 max(0, cols.min() - mc):min(w, cols.max() + mc + 1)]


class PatchDataset(Dataset):
    """Patches (uint8) -> tenseurs redimensionnés pour le modèle visé.
    size=32 pour MLP/LeNet-5, 224 pour AlexNet/VGG-16/ResNet (convention ImageNet).
    `noise_std` > 0 : bruit gaussien en entrée (autoencodeur débruiteur)."""

    def __init__(self, patches: np.ndarray, labels: np.ndarray, size: int = 224,
                 augment: bool = False, normalize_imagenet: bool = True):
        self.patches, self.labels, self.size = patches, labels, size
        self.augment, self.norm = augment, normalize_imagenet

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, i: int):
        p = self.patches[i]
        if self.augment and np.random.random() < 0.5:
            p = p[:, ::-1]
        if p.shape[0] != self.size:
            p = np.asarray(Image.fromarray(np.ascontiguousarray(p)).resize((self.size, self.size), Image.BILINEAR))
        x = normalize(p) if self.norm else (p.astype(np.float32) / 255.0).transpose(2, 0, 1).copy()
        return torch.from_numpy(x), int(self.labels[i])


class CorridorDataset(Dataset):
    """BO3 : (image normalisée, cibles [occupation, libre gauche, libre droite]) à la taille `size_wh`.
    Les cibles sont recalculées sur le masque APRÈS augmentation (un flip échange gauche et droite)."""

    def __init__(self, images: np.ndarray, masks: np.ndarray, size_wh=(256, 144), augment: bool = False,
                 repeats: int = 1):
        from .corridor import corridor_mask
        self.images, self.masks, self.size, self.augment, self.repeats = images, masks, tuple(size_wh), augment, repeats
        self.corridor = corridor_mask(size_wh[1], size_wh[0])

    def __len__(self) -> int:
        return len(self.images) * self.repeats

    def __getitem__(self, i: int):
        from .corridor import corridor_targets
        j = i % len(self.images)
        img, m = self.images[j], self.masks[j]
        if self.augment:
            rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
            img, m = augment_pair(img, m, rng, scale_range=(0.8, 1.0))
        img = np.asarray(Image.fromarray(img).resize(self.size, Image.BILINEAR))
        m = np.asarray(Image.fromarray(m.astype(np.uint8)).resize(self.size, Image.NEAREST))
        return torch.from_numpy(normalize(img)), torch.from_numpy(corridor_targets(m, self.corridor))
