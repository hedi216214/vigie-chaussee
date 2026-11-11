"""
config.py — Configuration centrale du projet RoadAnomaly (BO1 / BO2 / BO3).

Toutes les valeurs numériques de ce fichier sont des HYPERPARAMÈTRES ou des SEUILS MÉTIER
choisis a priori. Ce ne sont PAS des résultats : elles doivent être ajustées / justifiées
par l'étudiant à partir des expériences réelles (validation uniquement, jamais le test).
"""
from __future__ import annotations

import json
import os
import platform
import random
from dataclasses import asdict, dataclass, field

import numpy as np

# URL officielle de téléchargement du dataset (EPFL CVLab). À vérifier si le lien change.
DATASET_URL = "https://datasets-cvlab.epfl.ch/2019-road-anomaly/RoadAnomaly.zip"

# Poids Cityscapes de DeepLabV3+ (dépôt VainF/DeepLabV3Plus-Pytorch, README consulté).
# Les liens sont ceux du README ; vérifiez qu'ils sont toujours actifs avant l'exécution.
VAINF_REPO = "https://github.com/VainF/DeepLabV3Plus-Pytorch.git"
CITYSCAPES_CKPTS = {
    # MobileNetV2, output stride 16 (Dropbox, téléchargement direct avec ?dl=1)
    "deeplabv3plus_mobilenet": {
        "file": "best_deeplabv3plus_mobilenet_cityscapes_os16.pth",
        "url": "https://www.dropbox.com/s/753ojyvsh3vdjol/best_deeplabv3plus_mobilenet_cityscapes_os16.pth?dl=1",
        "output_stride": 16,
    },
    # ResNet-101, output stride 16 (Google Drive -> téléchargement avec gdown)
    "deeplabv3plus_resnet101": {
        "file": "best_deeplabv3plus_resnet101_cityscapes_os16.pth",
        "gdrive_id": "1t7TC8mxQaFECt4jutdq_NMnWxdm6B-Nb",
        "output_stride": 16,
    },
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ProjectPaths:
    """Chemins paramétrables (Colab, Drive ou local)."""
    project_dir: str
    data_dir: str | None = None      # dossier qui contient RoadAnomaly/ (par défaut <project>/data)
    outputs_name: str = "outputs"    # "outputs_quick" en mode test rapide : les vrais résultats ne sont pas écrasés

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = os.path.join(self.project_dir, "data")

    @property
    def dataset_root(self) -> str:
        return os.path.join(self.data_dir, "RoadAnomaly")

    @property
    def outputs(self) -> str:
        return os.path.join(self.project_dir, self.outputs_name)

    def sub(self, name: str) -> str:
        """Sous-dossier de outputs/ (créé si besoin) : figures, results, checkpoints, splits, weights…"""
        path = os.path.join(self.outputs, name)
        os.makedirs(path, exist_ok=True)
        return path


@dataclass
class TrainConfig:
    """Hyperparamètres par défaut (point de départ, à ajuster sur la validation)."""
    seed: int = 42
    work_size: tuple = (512, 288)        # (largeur, hauteur), 16:9 comme les images 1280×720
    batch_size: int = 4
    epochs: int = 60
    patience: int = 15                    # early stopping sur la métrique de validation
    lr_backbone: float = 1e-4             # couches pré-entraînées
    lr_head: float = 1e-3                 # couches nouvelles
    weight_decay: float = 1e-4
    max_pos_weight: float = 10.0          # plafond du poids des pixels anormaux dans la BCE
    amp: bool = True                      # précision mixte sur GPU
    num_workers: int = 2
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Fixe toutes les graines. `deterministic=True` rend cuDNN déterministe (plus lent)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    except ImportError:  # modules NumPy utilisables sans PyTorch
        pass


def get_device():
    """Renvoie cuda si disponible, sinon cpu (avec un avertissement explicite)."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("[ATTENTION] Aucun GPU détecté : l'entraînement sera très lent. "
          "Dans Colab : Exécution > Modifier le type d'exécution > GPU.")
    return torch.device("cpu")


def environment_report() -> dict:
    """Versions des bibliothèques, à enregistrer avec chaque résultat (reproductibilité)."""
    report = {"python": platform.python_version(), "numpy": np.__version__}
    for mod in ["torch", "torchvision", "sklearn", "PIL", "gradio"]:
        try:
            m = __import__(mod)
            report[mod] = getattr(m, "__version__", "?")
        except ImportError:
            report[mod] = None
    try:
        import torch
        report["cuda"] = torch.version.cuda
        report["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return report


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
