"""
bo_models.py — les 9 modèles du projet (3 par objectif métier), commentés couche par couche.

Pour chaque BO : deux modèles développés FROM SCRATCH (poids initialisés au hasard, tout est appris sur
RoadAnomaly) et un modèle PRÉ-ENTRAÎNÉ (poids ImageNet réutilisés, puis ajustés : transfert
d'apprentissage). Le notebook 03 affiche le code de chaque classe avant de l'entraîner.

Notations des commentaires : (C, H, W) = canaux, hauteur, largeur d'une image dans le réseau.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(ci: int, co: int) -> nn.Sequential:
    """Brique de base des réseaux from scratch : deux convolutions 3×3 qui gardent la taille de l'image.
    Conv 3×3 (détecte des motifs locaux) -> BatchNorm (stabilise l'apprentissage) -> ReLU (non-linéarité),
    deux fois de suite."""
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                         nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


# ======================================================================================
# BO1 — Un obstacle est-il présent dans une zone de 96 × 96 pixels ?
# ======================================================================================
class SmallCNN(nn.Module):
    """BO1 · modèle 1 · FROM SCRATCH — CNN à 4 blocs (classification avec / sans obstacle).

    Entrée : patch RGB normalisé (3, 96, 96). Sortie : 2 scores (logits) « sans obstacle », « avec obstacle ».
    Idée : chaque bloc double le nombre de filtres et divise la taille par 2 ; les premiers blocs voient des
    bords et des textures, les derniers des formes plus grandes (pelage, cône, roue).
    """

    def __init__(self, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Sequential(conv_block(3, 32), nn.MaxPool2d(2)),     # (3, 96, 96)   -> (32, 48, 48)
            nn.Sequential(conv_block(32, 64), nn.MaxPool2d(2)),    # (32, 48, 48)  -> (64, 24, 24)
            nn.Sequential(conv_block(64, 128), nn.MaxPool2d(2)),   # (64, 24, 24)  -> (128, 12, 12)
            nn.Sequential(conv_block(128, 256), nn.MaxPool2d(2)),  # (128, 12, 12) -> (256, 6, 6)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),      # moyenne de chaque carte : (256, 6, 6) -> (256, 1, 1)
            nn.Flatten(),                 # -> vecteur de 256 valeurs
            nn.Dropout(dropout),          # éteint 30 % des valeurs à l'entraînement (contre le sur-apprentissage)
            nn.Linear(256, n_classes),    # couche dense de décision -> 2 scores
        )

    def forward(self, x):
        return self.head(self.features(x))


class ConvAE(nn.Module):
    """BO1 · modèle 2 · FROM SCRATCH — autoencodeur convolutionnel débruiteur (non supervisé).

    Entraîné UNIQUEMENT sur des patches SANS obstacle, avec du bruit gaussien en entrée (σ = 0,1) et le
    patch propre comme cible. Il apprend donc à reconstruire « la route normale ». Score d'anomalie d'un
    patch = erreur de reconstruction : un obstacle, jamais vu, devrait être mal reconstruit.
    Entrée / sortie : patch RGB dans [0, 1], (3, 96, 96).
    """

    def __init__(self, latent_ch: int = 64):
        super().__init__()

        def down(ci, co):   # convolution 4×4 de pas 2 : divise la taille par 2
            return nn.Sequential(nn.Conv2d(ci, co, 4, 2, 1), nn.BatchNorm2d(co), nn.LeakyReLU(0.2, True))

        def up(ci, co):     # convolution transposée 4×4 de pas 2 : multiplie la taille par 2
            return nn.Sequential(nn.ConvTranspose2d(ci, co, 4, 2, 1), nn.BatchNorm2d(co), nn.ReLU(True))

        self.encoder = nn.Sequential(          # compresse l'image
            down(3, 32),                       # (3, 96, 96)   -> (32, 48, 48)
            down(32, 64),                      # (32, 48, 48)  -> (64, 24, 24)
            down(64, 128),                     # (64, 24, 24)  -> (128, 12, 12)
            nn.Conv2d(128, latent_ch, 1),      # goulot d'étranglement : (64, 12, 12), 9 216 valeurs au lieu de 27 648
        )
        self.decoder = nn.Sequential(          # reconstruit l'image à partir du code compressé
            nn.Conv2d(latent_ch, 128, 1),      # (64, 12, 12)  -> (128, 12, 12)
            up(128, 64),                       # -> (64, 24, 24)
            up(64, 32),                        # -> (32, 48, 48)
            nn.ConvTranspose2d(32, 3, 4, 2, 1),  # -> (3, 96, 96)
            nn.Sigmoid(),                      # valeurs ramenées dans [0, 1], comme l'image d'entrée
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ResNet50Classifier(nn.Module):
    """BO1 · modèle 3 · PRÉ-ENTRAÎNÉ — ResNet-50 ImageNet (torchvision, poids IMAGENET1K_V2).

    Le ResNet-50 a appris sur 1,2 million de photos ImageNet à reconnaître 1 000 classes d'objets. On garde
    tout son extracteur de caractéristiques (50 couches, blocs résiduels) et on remplace seulement la
    dernière couche (1 000 classes) par une couche à 2 sorties. Entrée : patch agrandi à (3, 224, 224),
    normalisé comme ImageNet. Fine-tuning : taux d'apprentissage 1e-4 pour les couches pré-entraînées,
    1e-3 pour la nouvelle couche.
    """

    def __init__(self, n_classes: int = 2):
        super().__init__()
        import torchvision.models as tvm
        self.backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)   # (3, 224, 224) -> (2048,)
        self.backbone.fc = nn.Linear(2048, n_classes)   # nouvelle tête : 2048 caractéristiques -> 2 scores

    def forward(self, x):
        return self.backbone(x)


# ======================================================================================
# BO2 — Où est l'obstacle, pixel par pixel ? (segmentation, image 512 × 288)
# ======================================================================================
class UNet(nn.Module):
    """BO2 · modèle 1 · FROM SCRATCH — U-Net (Ronneberger et al., 2015).

    Forme en « U » : un encodeur réduit l'image pour comprendre CE QU'IL Y A, un décodeur la ré-agrandit pour
    dire OÙ ; des connexions de saut recopient les détails fins de l'encodeur vers le décodeur.
    Entrée : (3, 288, 512) normalisée. Sortie : (1, 288, 512), un score « obstacle » par pixel (logit).
    """

    def __init__(self, out_ch: int = 1, base: int = 32, depth: int = 4, dropout: float = 0.2):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]                 # [32, 64, 128, 256, 512]
        # Encodeur : (32, 288, 512) -> (64, 144, 256) -> (128, 72, 128) -> (256, 36, 64) -> (512, 18, 32)
        self.enc = nn.ModuleList([conv_block(3 if i == 0 else chs[i - 1], chs[i]) for i in range(depth + 1)])
        self.pool = nn.MaxPool2d(2)                                      # divise la taille par 2 entre deux niveaux
        # Décodeur : convolution transposée (× 2), concaténation avec le niveau correspondant, conv_block
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([conv_block(chs[i] * 2, chs[i]) for i in reversed(range(depth))])
        self.drop = nn.Dropout2d(dropout)                                # régularisation au fond du U
        self.head = nn.Conv2d(chs[0], out_ch, 1)                         # (32, 288, 512) -> (1, 288, 512)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.enc):          # descente : on garde chaque niveau pour les sauts
            x = block(x if i == 0 else self.pool(x))
            skips.append(x)
        x = self.drop(skips.pop())                     # fond du U : (512, 18, 32)
        for up, dec in zip(self.up, self.dec):         # remontée : agrandir, coller les détails, affiner
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        return self.head(x)


class UpBlock(nn.Module):
    """Étape du décodeur de ResNetUNet : agrandit × 2 (interpolation bilinéaire), concatène la carte de
    l'encodeur de même taille (connexion de saut), puis conv_block."""

    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.conv = conv_block(cin + cskip, cout)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1) if skip is not None else x)


class ResNetUNet(nn.Module):
    """BO2 · modèles 2 et 3 — U-Net à encodeur ResNet-34.

    pretrained=False -> modèle 2, FROM SCRATCH : l'encodeur ResNet-34 part de poids aléatoires.
    pretrained=True  -> modèle 3, PRÉ-ENTRAÎNÉ : l'encodeur reprend les poids ImageNet ; le décodeur est neuf.
    Seule l'initialisation diffère entre les deux : l'écart de résultat mesure l'apport du pré-entraînement.
    Entrée : (3, 288, 512) normalisée. Sortie : (1, 288, 512), score « obstacle » par pixel.
    """

    def __init__(self, pretrained: bool = True, out_ch: int = 1):
        super().__init__()
        import torchvision
        r = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder = nn.ModuleDict({                 # les couches du ResNet-34 (sans sa tête de classification)
            "stem": nn.Sequential(r.conv1, r.bn1, r.relu),   # conv 7×7 pas 2 : (3, 288, 512) -> (64, 144, 256)
            "pool": r.maxpool,                               # -> (64, 72, 128)
            "l1": r.layer1,                                  # 3 blocs résiduels -> (64, 72, 128)
            "l2": r.layer2,                                  # 4 blocs -> (128, 36, 64)
            "l3": r.layer3,                                  # 6 blocs -> (256, 18, 32)
            "l4": r.layer4,                                  # 3 blocs -> (512, 9, 16)
        })
        self.up4 = UpBlock(512, 256, 256)   # (512, 9, 16)   + saut l3 -> (256, 18, 32)
        self.up3 = UpBlock(256, 128, 128)   # + saut l2 -> (128, 36, 64)
        self.up2 = UpBlock(128, 64, 64)     # + saut l1 -> (64, 72, 128)
        self.up1 = UpBlock(64, 64, 64)      # + saut stem -> (64, 144, 256)
        self.up0 = UpBlock(64, 0, 32)       # sans saut -> (32, 288, 512)
        self.head = nn.Conv2d(32, out_ch, 1)  # -> (1, 288, 512)

    def forward(self, x):
        e = self.encoder
        s0 = e["stem"](x); s1 = e["l1"](e["pool"](s0)); s2 = e["l2"](s1); s3 = e["l3"](s2); s4 = e["l4"](s3)
        d = self.up1(self.up2(self.up3(self.up4(s4, s3), s2), s1), s0)
        return self.head(self.up0(d))


# ======================================================================================
# BO3 — Quelle part du couloir de conduite est occupée, reste-t-il de la place à gauche / à droite ?
# ======================================================================================
class CorridorCNN(nn.Module):
    """BO3 · modèle 1 · FROM SCRATCH — CNN régresseur.

    Entrée : image entière (3, 144, 256) normalisée. Sortie : 3 nombres entre 0 et 1 :
    [occupation du couloir, espace libre à gauche, espace libre à droite].
    """

    def __init__(self, n_out: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Sequential(conv_block(3, 32), nn.MaxPool2d(2)),      # (3, 144, 256)  -> (32, 72, 128)
            nn.Sequential(conv_block(32, 64), nn.MaxPool2d(2)),     # -> (64, 36, 64)
            nn.Sequential(conv_block(64, 128), nn.MaxPool2d(2)),    # -> (128, 18, 32)
            nn.Sequential(conv_block(128, 128), nn.MaxPool2d(2)),   # -> (128, 9, 16)
            nn.Sequential(conv_block(128, 256), nn.MaxPool2d(2)),   # -> (256, 4, 8)
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),   # -> 256 valeurs
                                  nn.Dropout(0.3), nn.Linear(256, n_out))   # -> 3 valeurs

    def forward(self, x):
        return torch.sigmoid(self.head(self.features(x)))   # sigmoïde : sorties dans [0, 1] (des proportions)


class CorridorMLP(nn.Module):
    """BO3 · modèle 2 · FROM SCRATCH — réseau de neurones dense (MLP, « ANN » du cours).

    Aucune convolution : l'image réduite à 64 × 36 est mise à plat en un vecteur de 6 912 valeurs, puis
    traversée par deux couches denses. Sert de point de comparaison : sans structure spatiale, le réseau a
    beaucoup de mal à localiser l'obstacle.
    """

    def __init__(self, size_wh=(64, 36), n_out: int = 3, hidden=(512, 128), dropout: float = 0.3):
        super().__init__()
        self.size = (size_wh[1], size_wh[0])
        self.net = nn.Sequential(
            nn.Flatten(),                                     # (3, 36, 64) -> 6 912 valeurs
            nn.Linear(3 * size_wh[0] * size_wh[1], hidden[0]), nn.BatchNorm1d(hidden[0]),
            nn.ReLU(inplace=True), nn.Dropout(dropout),       # couche cachée 1 : 512 neurones
            nn.Linear(hidden[0], hidden[1]), nn.BatchNorm1d(hidden[1]),
            nn.ReLU(inplace=True), nn.Dropout(dropout),       # couche cachée 2 : 128 neurones
            nn.Linear(hidden[1], n_out),                      # sortie : 3 valeurs
        )

    def forward(self, x):
        x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)   # (3, 144, 256) -> (3, 36, 64)
        return torch.sigmoid(self.net(x))


class PretrainedRegressor(nn.Module):
    """BO3 · modèle 3 · PRÉ-ENTRAÎNÉ — ResNet-18 ImageNet, tête de régression à 3 sorties.

    Extracteur ResNet-18 (18 couches, poids ImageNet) conservé ; sa couche finale de 1 000 classes est
    remplacée par Dropout + couche dense 512 -> 3. Entrée (3, 144, 256) normalisée ; sortie : les trois
    proportions du couloir. Fine-tuning à taux différenciés (1e-4 / 1e-3).
    """

    def __init__(self, n_out: int = 3):
        super().__init__()
        import torchvision.models as tvm
        self.backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)   # -> 512 caractéristiques
        self.backbone.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, n_out))   # nouvelle tête

    def forward(self, x):
        return torch.sigmoid(self.backbone(x))


# ======================================================================================
# Outils
# ======================================================================================
def pretrained_params(model: nn.Module) -> tuple[list, list]:
    """Sépare (paramètres pré-entraînés, nouveaux paramètres) pour des taux d'apprentissage différenciés :
    tout ce qui est sous `encoder.` ou `backbone.` (sauf la nouvelle tête `backbone.fc`) est pré-entraîné."""
    pre, new = [], []
    for n, p in model.named_parameters():
        is_pre = n.startswith(("encoder.", "backbone.")) and not n.startswith("backbone.fc")
        (pre if is_pre else new).append(p)
    return pre, new


BUILDERS = {
    "bo1_cnn": lambda: SmallCNN(),
    "bo1_ae": lambda: ConvAE(),
    "bo1_resnet50": lambda: ResNet50Classifier(),
    "bo2_unet": lambda: UNet(),
    "bo2_resunet_scratch": lambda: ResNetUNet(pretrained=False),
    "bo2_resunet_imagenet": lambda: ResNetUNet(pretrained=True),
    "bo3_cnn": lambda: CorridorCNN(),
    "bo3_mlp": lambda: CorridorMLP(),
    "bo3_resnet18": lambda: PretrainedRegressor(),
}


def build_bo_model(model_id: str) -> nn.Module:
    """Construit un des 9 modèles par son identifiant (voir src/models_catalog.py)."""
    return BUILDERS[model_id]()


def model_summary(model: nn.Module, input_shape: tuple, depth: int = 2, pretrained: bool = False):
    """Résumé de l'architecture (comme `model.summary()` de Keras) : pour chaque couche jusqu'à `depth`
    niveaux, la forme de sa sortie et son nombre de paramètres. Renvoie un DataFrame pandas."""
    import pandas as pd
    rows, hooks = [], []

    def register(mod, name, level):
        def hook(_m, _i, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            n = sum(p.numel() for p in _m.parameters())
            rows.append({"couche": name, "type": type(_m).__name__,
                         "sortie": tuple(o.shape[1:]) if hasattr(o, "shape") else "-", "paramètres": n})
        hooks.append(mod.register_forward_hook(hook))
        if level < depth:
            for child_name, child in mod.named_children():
                register(child, f"{name}.{child_name}" if name else child_name, level + 1)

    for child_name, child in model.named_children():
        register(child, child_name, 1)
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        model(torch.zeros(1, *input_shape, device=dev))
    for h in hooks:
        h.remove()
    model.train(was_training)
    df = pd.DataFrame(rows)
    # les hooks se déclenchent dans l'ordre de fin d'exécution : on remet l'ordre de déclaration
    order = {n: i for i, n in enumerate(_names(model, depth))}
    df = df.drop_duplicates("couche").assign(_o=lambda d: d.couche.map(order)).sort_values("_o").drop(columns="_o")
    total = sum(p.numel() for p in model.parameters())
    pre, _ = pretrained_params(model)
    n_pre = sum(p.numel() for p in pre)
    print(f"Paramètres : {total:,} au total".replace(",", " "),
          (f"| dont {n_pre:,} repris d'ImageNet, {total - n_pre:,} nouveaux".replace(",", " ") if pretrained
           else "| tous initialisés au hasard et appris from scratch"))
    return df.reset_index(drop=True)


def _names(model, depth, prefix="", level=1):
    out = []
    for n, c in model.named_children():
        full = f"{prefix}.{n}" if prefix else n
        out.append(full)
        if level < depth:
            out += _names(c, depth, full, level + 1)
    return out
