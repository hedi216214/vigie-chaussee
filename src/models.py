"""
models.py — Architectures du projet, reliées aux familles vues en cours.

    ANN / MLP .................. MLPClassifier            (BO1, baseline présence/absence)
    CNN ........................ LeNet5, AlexNet, VGG-16, ResNet-18/50 (BO1, transfer learning)
    Autoencodeurs .............. VanillaAE, ConvAE (+ entraînement débruiteur) (BO1, non supervisé)
    Segmentation pré-entraînée . DeepLabV3+ ResNet-101 Cityscapes (BO2), DeepLabV3+ MobileNetV2
                                 Cityscapes (BO3, segmentation de la chaussée)
    RNN / LSTM / GRU ........... CNNRecurrent (extension exploratoire sur séquences simulées)

Les poids Cityscapes viennent du dépôt VainF/DeepLabV3Plus-Pytorch (voir config.CITYSCAPES_CKPTS).
Si ce dépôt ou ces poids sont indisponibles, `build_deeplabv3plus_fallback` fournit un
DeepLabV3+ avec encodeur ImageNet (segmentation_models_pytorch) : ce n'est PAS un modèle
Cityscapes, et cela doit être signalé dans le rapport.
"""
from __future__ import annotations

import os
import subprocess
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CITYSCAPES_CKPTS, VAINF_REPO

# Classes Cityscapes (trainIds 0..18) — ordre standard du benchmark.
CITYSCAPES_CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
                      "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
                      "truck", "bus", "train", "motorcycle", "bicycle"]


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


# --------------------------------------------------------------------------------------
# 1. ANN / MLP
# --------------------------------------------------------------------------------------
class MLPClassifier(nn.Module):
    """Perceptron multicouche sur le patch aplati (32×32×3 par défaut).
    Baseline : ignore la structure spatiale, sert de point de comparaison pour les CNN."""

    def __init__(self, in_dim: int = 32 * 32 * 3, hidden=(512, 128), n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        layers, d = [nn.Flatten()], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------------------
# 2. CNN classiques
# --------------------------------------------------------------------------------------
class LeNet5(nn.Module):
    """LeNet-5 (LeCun et al., 1998) adapté à une entrée RGB 32×32, entraîné de zéro."""

    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5), nn.Tanh(), nn.AvgPool2d(2),     # 32 -> 28 -> 14
            nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2),    # 14 -> 10 -> 5
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.Tanh(),
                                        nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_classifier(name: str, n_classes: int = 2, pretrained: bool = True,
                     freeze_backbone: bool = False) -> nn.Module:
    """AlexNet, VGG-16, ResNet-18 ou ResNet-50 (torchvision), tête remplacée pour n_classes.
    pretrained=True -> poids ImageNet (transfer learning) ; False -> entraînement de zéro.
    freeze_backbone=True -> seule la dernière couche est entraînée (feature extraction)."""
    import torchvision.models as tvm
    name = name.lower()
    if name == "alexnet":
        m = tvm.alexnet(weights=tvm.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None)
        head_parent, head_attr = m.classifier, "6"
    elif name == "vgg16":
        m = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        head_parent, head_attr = m.classifier, "6"
    elif name == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        head_parent, head_attr = m, "fc"
    elif name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        head_parent, head_attr = m, "fc"
    else:
        raise ValueError(f"Modèle inconnu : {name}")
    if freeze_backbone:
        for p in m.parameters():
            p.requires_grad = False
    old = getattr(head_parent, head_attr) if head_attr != "6" else head_parent[6]
    new = nn.Linear(old.in_features, n_classes)
    if head_attr == "6":
        head_parent[6] = new
    else:
        setattr(head_parent, head_attr, new)
    return m


def head_parameters(model: nn.Module) -> list:
    """Paramètres de la tête (couche remplacée) : reçoivent un taux d'apprentissage plus élevé."""
    if hasattr(model, "fc"):
        return list(model.fc.parameters())
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        return list(model.classifier[-1].parameters())
    return list(model.parameters())


# --------------------------------------------------------------------------------------
# 3. Autoencodeurs (détection non supervisée par erreur de reconstruction)
# --------------------------------------------------------------------------------------
class VanillaAE(nn.Module):
    """Autoencodeur dense : entrée [0,1] (3, S, S) aplatie, goulot de `latent` neurones."""

    def __init__(self, size: int = 64, latent: int = 128):
        super().__init__()
        d = 3 * size * size
        self.size = size
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(d, 1024), nn.ReLU(True), nn.Linear(1024, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, 1024), nn.ReLU(True), nn.Linear(1024, d), nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x)).view(-1, 3, self.size, self.size)


class ConvAE(nn.Module):
    """Autoencodeur convolutionnel (64×64 -> 8×8×latent -> 64×64).
    Version débruiteuse : même architecture, entraînée avec du bruit en entrée
    (train.train_autoencoder(noise_std > 0)) et la cible propre."""

    def __init__(self, latent_ch: int = 64):
        super().__init__()

        def down(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 4, 2, 1), nn.BatchNorm2d(co), nn.LeakyReLU(0.2, True))

        def up(ci, co):
            return nn.Sequential(nn.ConvTranspose2d(ci, co, 4, 2, 1), nn.BatchNorm2d(co), nn.ReLU(True))

        self.encoder = nn.Sequential(down(3, 32), down(32, 64), down(64, 128), nn.Conv2d(128, latent_ch, 1))
        self.decoder = nn.Sequential(nn.Conv2d(latent_ch, 128, 1), up(128, 64), up(64, 32),
                                     nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))


def reconstruction_error(model: nn.Module, x: torch.Tensor, per_pixel: bool = False) -> torch.Tensor:
    """Erreur quadratique par échantillon (score d'anomalie) ou carte d'erreur (B, H, W)."""
    with torch.no_grad():
        err = (model(x) - x) ** 2
    return err.mean(1) if per_pixel else err.flatten(1).mean(1)


# --------------------------------------------------------------------------------------
# 4. DeepLabV3+ pré-entraîné Cityscapes (BO2, BO3)
# --------------------------------------------------------------------------------------
def ensure_vainf_repo(repo_dir: str) -> str:
    """Clone le dépôt VainF/DeepLabV3Plus-Pytorch si absent et l'ajoute au sys.path."""
    if not os.path.isdir(os.path.join(repo_dir, "network")):
        subprocess.run(["git", "clone", "--depth", "1", VAINF_REPO, repo_dir], check=True)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    return repo_dir


def download_cityscapes_ckpt(arch: str, weights_dir: str) -> str:
    """Télécharge le checkpoint Cityscapes listé dans le README VainF. Lève une erreur claire
    si le lien ne répond plus (dans ce cas, téléchargez-le à la main dans weights_dir)."""
    info = CITYSCAPES_CKPTS[arch]
    os.makedirs(weights_dir, exist_ok=True)
    path = os.path.join(weights_dir, info["file"])
    if os.path.isfile(path) and os.path.getsize(path) > 1_000_000:
        return path
    try:
        if "url" in info:
            import urllib.request
            urllib.request.urlretrieve(info["url"], path)
        else:
            import gdown  # pip install gdown
            gdown.download(id=info["gdrive_id"], output=path, quiet=False)
    except Exception as e:
        raise RuntimeError(f"Téléchargement impossible pour {arch} ({e}). Placez {info['file']} "
                           f"manuellement dans {weights_dir}.") from e
    if not os.path.isfile(path) or os.path.getsize(path) < 1_000_000:
        raise RuntimeError(f"Fichier {path} absent ou trop petit : lien probablement expiré.")
    return path


def build_vainf_deeplabv3plus(arch: str, repo_dir: str, ckpt_path: str | None,
                              num_classes: int = 19) -> nn.Module:
    """arch = 'deeplabv3plus_resnet101' (BO2) ou 'deeplabv3plus_mobilenet' (BO3).
    Chargement conforme au README VainF : torch.load(...)['model_state']."""
    ensure_vainf_repo(repo_dir)
    import network  # module du dépôt VainF
    fn = network.modeling.__dict__[arch]
    os_ = CITYSCAPES_CKPTS.get(arch, {}).get("output_stride", 16)
    try:  # pretrained_backbone=False : inutile de télécharger ImageNet, le checkpoint contient tout
        model = fn(num_classes=num_classes, output_stride=os_, pretrained_backbone=False)
    except TypeError:
        model = fn(num_classes=num_classes, output_stride=os_)
    if ckpt_path:
        # weights_only=False : le checkpoint contient aussi l'état de l'optimiseur.
        # À n'utiliser qu'avec un fichier de source connue (ici le dépôt VainF).
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model_state", state))
    return model


def build_deeplabv3plus_fallback(encoder: str = "resnet101", classes: int = 1) -> nn.Module:
    """Repli SANS poids Cityscapes : DeepLabV3+ (segmentation_models_pytorch), encodeur ImageNet.
    encoder = 'resnet101' (BO2) ou 'mobilenet_v2' (BO3). À mentionner comme écart au protocole."""
    import segmentation_models_pytorch as smp
    return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights="imagenet", classes=classes)


def replace_last_conv(model: nn.Module, out_channels: int = 1) -> nn.Module:
    """Remplace la dernière Conv2d de classification (19 classes Cityscapes) par une Conv2d à
    `out_channels` sorties : fine-tuning binaire anomalie / normal (BO2)."""
    last_name, last = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            last_name, last = name, mod
    if last is None:
        raise ValueError("Aucune Conv2d trouvée dans le modèle.")
    parent = model.get_submodule(last_name.rsplit(".", 1)[0]) if "." in last_name else model
    attr = last_name.rsplit(".", 1)[-1]
    new = nn.Conv2d(last.in_channels, out_channels, last.kernel_size, last.stride, last.padding,
                    bias=last.bias is not None)
    if isinstance(parent, nn.Sequential):
        parent[int(attr)] = new
    else:
        setattr(parent, attr, new)
    return model


class ResizeOutput(nn.Module):
    """Garantit une sortie à la résolution de l'entrée (B, C, H, W)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        y = self.model(x)
        if isinstance(y, dict):  # convention torchvision
            y = y["out"]
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return y


def ood_scores(logits: torch.Tensor) -> dict:
    """Scores d'anomalie « zero-shot » à partir des logits Cityscapes (B, 19, H, W) :
    plus le score est élevé, plus le pixel est jugé hors distribution.
        msp     = 1 - max softmax   (Hendrycks & Gimpel, 2017)
        entropy = entropie normalisée de la softmax
        maxlogit= - max logit       (Hendrycks et al., 2022)
    """
    p = logits.float().softmax(1)
    msp = 1 - p.max(1).values
    ent = -(p * p.clamp_min(1e-12).log()).sum(1) / torch.log(torch.tensor(float(p.shape[1])))
    maxlogit = -logits.float().max(1).values
    return {"msp": msp, "entropy": ent, "maxlogit": maxlogit}


# --------------------------------------------------------------------------------------
# 5. CNN + RNN (extension exploratoire)
# --------------------------------------------------------------------------------------
class CNNRecurrent(nn.Module):
    """Encodeur CNN (ResNet-18 ImageNet, gelé par défaut) appliqué à chaque image d'une
    séquence, suivi d'un LSTM ou d'un GRU. Entrée (B, T, 3, H, W), sortie (B, n_out).
    Sur RoadAnomaly (images fixes), les séquences sont SIMULÉES : résultats exploratoires."""

    def __init__(self, rnn: str = "lstm", hidden: int = 128, n_out: int = 1, freeze_encoder: bool = True):
        super().__init__()
        import torchvision.models as tvm
        r = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(r.children())[:-1])  # -> (N, 512, 1, 1)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn.lower()]
        self.rnn = cls(512, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def encode(self, x):
        """(B, T, 3, H, W) -> caractéristiques par image (B, T, 512)."""
        b, t = x.shape[:2]
        return self.encoder(x.flatten(0, 1)).flatten(1).view(b, t, -1)

    def forward_features(self, f):
        """À partir de caractéristiques pré-calculées (B, T, 512) : utile quand l'encodeur est gelé."""
        out, _ = self.rnn(f)
        return self.head(out[:, -1])

    def forward(self, x):
        return self.forward_features(self.encode(x))


# ======================================================================================
# 6. Catalogue retenu : pour chaque BO, 2 modèles développés from scratch + 1 pré-entraîné
# ======================================================================================


# Les 9 modèles du projet (3 par BO) sont définis et commentés dans src/bo_models.py.
from .bo_models import (conv_block as _conv_block, SmallCNN, ConvAE as _ConvAE_BO1, ResNet50Classifier, UNet,  # noqa: E402,F401
                        UpBlock as _UpBlock, ResNetUNet, CorridorCNN, CorridorMLP, PretrainedRegressor,
                        pretrained_params, build_bo_model, model_summary)
from .models_catalog import BO_MODELS  # noqa: E402,F401
