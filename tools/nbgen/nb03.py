import ast
import os

from common import INTEGRITY, code, load_split_cells, md, setup_cells, write

# Le code de chaque modèle est recopié tel quel depuis src/bo_models.py (source unique) : le notebook
# affiche et exécute exactement la classe que les notebooks 04 et 05 rechargent.
_SRC = open(os.path.join(os.path.dirname(__file__), "..", "..", "src", "bo_models.py"), encoding="utf-8").read()
_TREE = ast.parse(_SRC)


def src_of(*names):
    parts = []
    for node in _TREE.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            parts.append(ast.get_source_segment(_SRC, node))
    assert len(parts) == len(names), names
    return code("\n\n\n".join(parts))


def model_section(num, title, intro, cls_cells, summary, train_md, train_code):
    return [md(f"### {num} {title}\n\n{intro.strip()}"),
            md("**Définition du modèle** (code commenté couche par couche) :"),
            *cls_cells,
            md("**Résumé de l'architecture** : forme de la sortie et nombre de paramètres de chaque couche."),
            code(summary),
            md(f"**Entraînement.** {train_md.strip()}"),
            code(train_code)]


nb03 = [md("""
# 03 — Modélisation : trois modèles par objectif

Pour chaque objectif métier (BO), on construit **deux modèles développés from scratch** (tous les poids
partent du hasard et sont appris sur RoadAnomaly) et **un modèle pré-entraîné** (on réutilise un réseau
déjà entraîné sur ImageNet, 1,2 million de photos, puis on l'ajuste : transfert d'apprentissage).
Les neuf modèles sont entraînés sur les mêmes données, préparées au notebook 02. Ce notebook entraîne et
sauvegarde ; la comparaison et le choix du meilleur modèle se font au notebook 04.

| BO | Question | Modèle 1 · from scratch | Modèle 2 · from scratch | Modèle 3 · pré-entraîné |
|---|---|---|---|---|
| BO1 | Un obstacle est-il présent dans cette zone ? | CNN à 4 blocs | Autoencodeur débruiteur | ResNet-50 ImageNet |
| BO2 | Où est l'obstacle, pixel par pixel ? | U-Net | U-Net à encodeur ResNet-34 (poids aléatoires) | U-Net à encodeur ResNet-34 ImageNet |
| BO3 | Quelle part de la voie est occupée, où passer ? | CNN régresseur | Réseau dense (MLP) | ResNet-18 ImageNet |
"""), md("""
### Étapes de ce notebook

1. Importer les bibliothèques et préparer les outils communs
2. BO1 — préparer les zones, puis définir, résumer et entraîner les 3 modèles
3. BO2 — préparer les images, puis définir, résumer et entraîner les 3 modèles
4. BO3 — préparer les cibles du couloir, puis définir, résumer et entraîner les 3 modèles
5. Récapituler l'entraînement
"""), INTEGRITY] + setup_cells() + load_split_cells() + [
    md("""
## 1. Importer les bibliothèques et préparer les outils communs

- `time_per_item` mesure le temps de calcul d'une prédiction (en millisecondes par zone ou par image).
- `record` enregistre le nombre de paramètres, la meilleure époque, la durée, l'historique, sauvegarde
  les poids et trace les courbes d'apprentissage.
- `model_summary` affiche l'architecture, comme `model.summary()` dans Keras.
"""),
    code("""
    import time, torch, torch.nn as nn, torch.nn.functional as F, pandas as pd, matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from src.config import get_device
    from src.dataset import PatchDataset, SegmentationDataset, CorridorDataset, normalize
    from src.models_catalog import BO_MODELS
    from src.bo_models import model_summary, pretrained_params
    from src.models import count_parameters
    from src.train import (train_classifier, predict_proba, train_autoencoder, train_segmentation, predict_masks,
                           train_regressor, predict_regression, save_checkpoint)
    from src.visualize import plot_learning_curves

    device = get_device()
    print('Calcul sur :', device)
    CKPT = paths.sub('checkpoints')
    SUMMARY = {}

    @torch.no_grad()
    def time_per_item(model, x, n=20):
        model.eval().to(device); x = x.to(device)
        for _ in range(3): model(x)
        if device.type == 'cuda': torch.cuda.synchronize()
        t = time.time()
        for _ in range(n): model(x)
        if device.type == 'cuda': torch.cuda.synchronize()
        return (time.time() - t) / n / len(x) * 1000      # ms par élément

    def record(mid, model, hist, ms):
        SUMMARY[mid] = {'params_M': count_parameters(model) / 1e6, 'best_epoch': hist.get('best_epoch'),
                        'train_seconds': hist.get('seconds'), 'ms_per_item': ms,
                        'history': {k: v for k, v in hist.items() if isinstance(v, list)}}
        save_checkpoint(os.path.join(CKPT, f'{mid}.pth'), model,
                        {'model_id': mid, **{k: v for k, v in SUMMARY[mid].items() if k != 'history'}})
        keys = [k for k, v in hist.items() if isinstance(v, list)]
        plot_learning_curves(hist, keys, mid, path=os.path.join(FIG, f'03_curves_{mid}.png')); plt.show()
        print(f"{mid} : meilleure époque {hist.get('best_epoch')}, {SUMMARY[mid]['params_M']:.2f} M paramètres, "
              f"{ms:.2f} ms par élément")
    """),
    md("""
La brique `conv_block` sert à tous les modèles from scratch : deux convolutions 3 × 3, chacune suivie
d'une normalisation (BatchNorm) et d'une activation ReLU. Elle garde la taille de l'image et change le
nombre de canaux.
"""),
    src_of("conv_block"),

    # ---------------------------------------------------------------------------------- BO1
    md("""
## 2. BO1 — Détecter la présence d'un obstacle dans une zone

### 2.1 Préparer les entrées

Les zones de 96 × 96 du notebook 02 : 1 527 en entraînement, dont environ 40 % avec obstacle. La perte de
classification est pondérée (les zones avec obstacle, moins nombreuses, comptent davantage). À
l'entraînement, chaque zone est retournée horizontalement une fois sur deux (miroir).
"""),
    code("""
    zp = np.load(os.path.join(paths.sub('processed'), 'patches_96.npz'))
    P = {s: (zp[f'X_{s}'], zp[f'y_{s}']) for s in ['train', 'val', 'test']}
    for s, (X, y) in P.items():
        print(f'{s:5s} {X.shape}  avec obstacle {int(y.sum())}  sans obstacle {int((y == 0).sum())}')
    n_pos, n_neg = int(P['train'][1].sum()), int((P['train'][1] == 0).sum())
    class_weights = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32)

    def patch_loaders(size, bs, norm=True, subset=None):
        out = {}
        for s, (X, y) in P.items():
            if subset is not None and s in subset:
                keep = subset[s](y); X, y = X[keep], y[keep]
            out[s] = DataLoader(PatchDataset(X, y, size=size, augment=(s == 'train'), normalize_imagenet=norm),
                                batch_size=bs, shuffle=(s == 'train'), drop_last=(s == 'train'), num_workers=cfg.num_workers)
        return out

    def save_bo1_scores(mid, val, test):   # scores de validation et de test, relus par le notebook 04
        np.savez_compressed(os.path.join(PREDS, f'{mid}.npz'), y_val=P['val'][1], s_val=val, y_test=P['test'][1], s_test=test)
    """),
] + model_section(
    "2.2", "Modèle 1 — CNN à 4 blocs (from scratch)",
    """
Réseau convolutif classique conçu pour ce projet. Quatre blocs `conv_block` + MaxPool : à chaque bloc,
le nombre de filtres double (32, 64, 128, 256) et la taille de l'image est divisée par deux
(96 → 48 → 24 → 12 → 6). Une moyenne globale résume chaque carte en un nombre, puis une couche dense
donne deux scores : « sans obstacle » et « avec obstacle ».

| Couche | Sortie | Rôle |
|---|---|---|
| Entrée | 3 × 96 × 96 | zone RGB normalisée |
| Bloc 1 à 4 (conv 3×3 ×2, BatchNorm, ReLU, MaxPool) | 32×48×48 → 64×24×24 → 128×12×12 → 256×6×6 | motifs de plus en plus grands |
| Moyenne globale + Dropout 0,3 | 256 | résumé de la zone, régularisation |
| Dense | 2 | décision avec / sans obstacle |
""",
    [src_of("SmallCNN")],
    """
    set_seed(cfg.seed)
    m = SmallCNN()
    model_summary(m, (3, 96, 96), depth=2)
    """,
    "40 époques au plus, arrêt anticipé si le F1 de validation ne progresse plus pendant 10 époques ; "
    "optimiseur AdamW, taux d'apprentissage 1e-3 ; perte d'entropie croisée pondérée.",
    """
    L = patch_loaders(96, 64)
    hist = train_classifier(m, L['train'], L['val'], device, epochs=EP(40), patience=10, lr_head=1e-3,
                            class_weights=class_weights, amp=cfg.amp)
    sv, _ = predict_proba(m, L['val'], device); st, _ = predict_proba(m, L['test'], device)
    save_bo1_scores('bo1_cnn', sv, st)
    record('bo1_cnn', m, hist, time_per_item(m, next(iter(L['test']))[0]))
    """) + model_section(
    "2.3", "Modèle 2 — Autoencodeur convolutionnel débruiteur (from scratch, non supervisé)",
    """
Un autoencodeur apprend à **reconstruire** son entrée en la faisant passer par un goulot d'étranglement.
On ne lui montre **que des zones sans obstacle** : il apprend à quoi ressemble une route normale.
Version débruiteuse : on ajoute du bruit à l'entrée et on lui demande de rendre la zone propre, ce qui
l'oblige à apprendre la structure plutôt qu'à recopier. À l'usage, le score d'anomalie d'une zone est son
erreur de reconstruction : un obstacle, jamais vu, devrait être mal reconstruit.

| Partie | Couches | Sortie |
|---|---|---|
| Encodeur | 3 convolutions 4×4 de pas 2 (+ BatchNorm, LeakyReLU), conv 1×1 | 3×96×96 → 64×12×12 |
| Décodeur | conv 1×1, 3 convolutions transposées 4×4 de pas 2, sigmoïde | 64×12×12 → 3×96×96 |
""",
    [src_of("ConvAE")],
    """
    set_seed(cfg.seed)
    m = ConvAE()
    model_summary(m, (3, 96, 96), depth=2)
    """,
    "Uniquement sur les zones sans obstacle ; bruit gaussien σ = 0,1 ; perte = erreur quadratique entre la "
    "reconstruction et la zone propre ; 60 époques au plus, patience 10.",
    """
    normal = {'train': lambda y: y == 0, 'val': lambda y: y == 0}
    Ln = patch_loaders(96, 64, norm=False, subset=normal)          # zones normales uniquement

    @torch.no_grad()
    def recon_scores(model, loader):                               # erreur de reconstruction par zone
        model.eval(); out = []
        for x, _ in loader:
            x = x.to(device); out.append(((model(x) - x) ** 2).flatten(1).mean(1).cpu().numpy())
        return np.concatenate(out)

    hist = train_autoencoder(m, Ln['train'], Ln['val'], device, epochs=EP(60), patience=10, noise_std=0.1)
    La_eval = {s: DataLoader(PatchDataset(*P[s], size=96, normalize_imagenet=False), batch_size=128) for s in ['val', 'test']}
    save_bo1_scores('bo1_ae', recon_scores(m, La_eval['val']), recon_scores(m, La_eval['test']))
    record('bo1_ae', m, hist, time_per_item(m, next(iter(La_eval['test']))[0]))
    """) + model_section(
    "2.4", "Modèle 3 — ResNet-50 pré-entraîné sur ImageNet",
    """
**Transfert d'apprentissage.** Le ResNet-50 (He et al., 2016) a appris à reconnaître 1 000 catégories
d'objets sur ImageNet. Ses 49 premières couches savent déjà extraire des caractéristiques visuelles
générales (bords, textures, formes, parties d'animaux, de véhicules). On garde ces couches et on remplace
seulement la dernière (1 000 classes) par une couche à 2 sorties. Les blocs résiduels (raccourcis qui
additionnent l'entrée d'un bloc à sa sortie) permettent d'entraîner un réseau aussi profond.

| Partie | Origine | Sortie |
|---|---|---|
| Entrée | zone agrandie à 224 × 224, normalisée comme ImageNet | 3 × 224 × 224 |
| conv1 + 4 étages de blocs résiduels (3, 4, 6, 3 blocs) | **ImageNet**, ajustés | 2048 × 7 × 7 |
| Moyenne globale | — | 2048 |
| Dense 2048 → 2 | **nouvelle** | 2 |
""",
    [src_of("ResNet50Classifier")],
    """
    set_seed(cfg.seed)
    m = ResNet50Classifier()
    model_summary(m, (3, 224, 224), depth=2, pretrained=True)
    """,
    "Taux d'apprentissage différenciés : 1e-4 pour les couches ImageNet (on les ajuste doucement), 1e-3 pour "
    "la nouvelle couche ; 20 époques au plus, patience 6.",
    """
    L = patch_loaders(224, 32)
    hist = train_classifier(m, L['train'], L['val'], device, epochs=EP(20), patience=6, lr_backbone=1e-4, lr_head=1e-3,
                            head_params=pretrained_params(m)[1], class_weights=class_weights, amp=cfg.amp)
    sv, _ = predict_proba(m, L['val'], device); st, _ = predict_proba(m, L['test'], device)
    save_bo1_scores('bo1_resnet50', sv, st)
    record('bo1_resnet50', m, hist, time_per_item(m, next(iter(L['test']))[0]))
    del m; torch.cuda.empty_cache()
    """) + [
    # ---------------------------------------------------------------------------------- BO2
    md("""
## 3. BO2 — Localiser l'obstacle au pixel

### 3.1 Préparer les entrées

Images entières 512 × 288 normalisées ; chaque époque parcourt 4 fois les 40 images d'entraînement, avec
une augmentation différente à chaque fois. Perte commune aux trois modèles : entropie croisée binaire
pondérée (poids du notebook 02 sur les pixels d'obstacle) + perte Dice (qui mesure directement le
recouvrement). Le meilleur état de chaque modèle est celui du meilleur Dice de validation.
"""),
    code("""
    train_dl = DataLoader(SegmentationDataset(DATA['train'][0], DATA['train'][1], augment=True, repeats=4),
                          batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers, pin_memory=True)

    def fit_bo2(mid, m, epochs, pretrained):
        set_seed(cfg.seed)
        head = pretrained_params(m)[1] if pretrained else None
        hist = train_segmentation(m, train_dl, DATA['val'][0], DATA['val'][1], device, epochs=EP(epochs),
                                  patience=cfg.patience, lr_backbone=1e-4, lr_head=1e-3, head_params=head,
                                  pos_weight=PREP['pos_weight'], amp=cfg.amp)
        pv, pt = predict_masks(m, DATA['val'][0], device), predict_masks(m, DATA['test'][0], device)
        np.savez_compressed(os.path.join(PREDS, f'{mid}.npz'), val=pv.astype(np.float16), test=pt.astype(np.float16))
        x = torch.from_numpy(np.stack([normalize(im) for im in DATA['test'][0][:2]]))
        record(mid, m, hist, time_per_item(m, x))
    """),
] + model_section(
    "3.2", "Modèle 1 — U-Net (from scratch)",
    """
Le U-Net (Ronneberger et al., 2015) est l'architecture de référence pour segmenter avec peu de données.
L'**encodeur** réduit l'image à chaque niveau pour comprendre ce qu'elle contient ; le **décodeur**
l'agrandit à nouveau pour dire où. Les **connexions de saut** recopient chaque niveau de l'encodeur dans
le décodeur, ce qui préserve les contours fins de l'obstacle.

| Niveau | Encodeur (sortie) | Décodeur (sortie) |
|---|---|---|
| 1 | 32 × 288 × 512 | 32 × 288 × 512 → couche finale 1 × 288 × 512 |
| 2 | 64 × 144 × 256 | 64 × 144 × 256 |
| 3 | 128 × 72 × 128 | 128 × 72 × 128 |
| 4 | 256 × 36 × 64 | 256 × 36 × 64 |
| fond du U | 512 × 18 × 32 (+ Dropout 0,2) | — |
""",
    [src_of("UNet")],
    """
    set_seed(cfg.seed)
    m = UNet()
    model_summary(m, (3, 288, 512), depth=1)
    """,
    "100 époques au plus, patience 15, AdamW 1e-3 avec décroissance cosinus du taux d'apprentissage.",
    """
    fit_bo2('bo2_unet', m, epochs=100, pretrained=False)
    del m; torch.cuda.empty_cache()
    """) + model_section(
    "3.3", "Modèle 2 — U-Net à encodeur ResNet-34, poids aléatoires (from scratch)",
    """
Même principe en U, mais l'encodeur est un **ResNet-34** (34 couches, blocs résiduels) et le décodeur
agrandit par interpolation bilinéaire. Ici l'encodeur part de **poids aléatoires** : tout est appris sur
nos 40 images. Ce modèle sert de témoin pour le modèle 3, qui a exactement la même architecture.

| Partie | Couches | Sortie |
|---|---|---|
| Encodeur ResNet-34 | conv 7×7 de pas 2, MaxPool, 4 étages résiduels (3, 4, 6, 3 blocs) | 64×144×256 → 64×72×128 → 128×36×64 → 256×18×32 → 512×9×16 |
| Décodeur | 5 étapes `UpBlock` (× 2, saut, conv_block) | 256×18×32 → … → 32×288×512 |
| Sortie | conv 1×1 | 1 × 288 × 512 |
""",
    [src_of("UpBlock", "ResNetUNet")],
    """
    set_seed(cfg.seed)
    m = ResNetUNet(pretrained=False)
    model_summary(m, (3, 288, 512), depth=1)
    """,
    "80 époques au plus, patience 15, AdamW 1e-3 pour tout le réseau.",
    """
    fit_bo2('bo2_resunet_scratch', m, epochs=80, pretrained=False)
    del m; torch.cuda.empty_cache()
    """) + model_section(
    "3.4", "Modèle 3 — U-Net à encodeur ResNet-34 pré-entraîné sur ImageNet",
    """
Exactement le réseau du modèle 2, mais l'encodeur reprend les **poids ImageNet** ; seul le décodeur part
de zéro. Comparer les modèles 2 et 3 isole l'effet du pré-entraînement : même architecture, mêmes
données, même graine, seule l'initialisation change.
""",
    [],
    """
    set_seed(cfg.seed)
    m = ResNetUNet(pretrained=True)
    model_summary(m, (3, 288, 512), depth=1, pretrained=True)
    """,
    "60 époques au plus, patience 15 ; taux 1e-4 pour l'encodeur ImageNet, 1e-3 pour le décodeur neuf.",
    """
    fit_bo2('bo2_resunet_imagenet', m, epochs=60, pretrained=True)
    del m; torch.cuda.empty_cache()
    """) + [
    # ---------------------------------------------------------------------------------- BO3
    md("""
## 4. BO3 — Occupation et espace libre du couloir de conduite

### 4.1 Préparer les entrées et les cibles

Le jeu de données n'annote pas la chaussée. On définit donc un **couloir de conduite fixe** devant le
véhicule : un trapèze du bas de l'image (de 10 % à 90 % de la largeur) jusqu'à 55 % de sa hauteur
(de 40 % à 60 %). Pour chaque image, trois cibles sont calculées à partir du vrai masque :

- **occupation** : part du couloir couverte par l'obstacle ;
- **libre à gauche / à droite** : espace libre de chaque côté de l'obstacle, en part de la largeur du
  couloir, mesuré au contact de l'obstacle avec la chaussée (25 % les plus bas de ses lignes).

Entrée des modèles : image 256 × 144. Les cibles sont recalculées après chaque augmentation (un miroir
échange la gauche et la droite). Perte Smooth L1 ; meilleur état = plus petite erreur moyenne de validation.
"""),
    code("""
    BO3_SIZE = (256, 144)
    ds = {s: CorridorDataset(DATA[s][0], DATA[s][1], size_wh=BO3_SIZE, augment=(s == 'train'), repeats=8 if s == 'train' else 1)
          for s in ['train', 'val', 'test']}
    loaders = {s: DataLoader(d, batch_size=8, shuffle=(s == 'train'), drop_last=(s == 'train'), num_workers=cfg.num_workers)
               for s, d in ds.items()}
    x0, t0 = ds['val'][0]
    print('entrée', tuple(x0.shape), '| cibles [occupation, libre gauche, libre droite] =', np.round(t0.numpy(), 3))

    def fit_bo3(mid, m, pretrained):
        set_seed(cfg.seed)
        head = pretrained_params(m)[1] if pretrained else None
        hist = train_regressor(m, loaders['train'], loaders['val'], device, epochs=EP(80), patience=15,
                               lr_backbone=1e-4, lr_head=1e-3, head_params=head)
        pv, tv = predict_regression(m, loaders['val'], device)
        pt, tt = predict_regression(m, loaders['test'], device)
        np.savez_compressed(os.path.join(PREDS, f'{mid}.npz'), p_val=pv, t_val=tv, p_test=pt, t_test=tt)
        record(mid, m, hist, time_per_item(m, next(iter(loaders['test']))[0]))
    """),
] + model_section(
    "4.2", "Modèle 1 — CNN régresseur (from scratch)",
    """
Même famille que le CNN de BO1, avec un bloc de plus car l'image est plus grande, et une sortie de
**régression** : trois nombres passés dans une sigmoïde pour rester entre 0 et 1 (ce sont des proportions).

| Couche | Sortie |
|---|---|
| Entrée | 3 × 144 × 256 |
| 5 blocs `conv_block` + MaxPool | 32×72×128 → 64×36×64 → 128×18×32 → 128×9×16 → 256×4×8 |
| Moyenne globale + Dropout 0,3 | 256 |
| Dense + sigmoïde | 3 : occupation, libre gauche, libre droite |
""",
    [src_of("CorridorCNN")],
    """
    set_seed(cfg.seed)
    m = CorridorCNN()
    model_summary(m, (3, 144, 256), depth=2)
    """,
    "80 époques au plus, patience 15, AdamW 1e-3.",
    """
    fit_bo3('bo3_cnn', m, pretrained=False)
    del m; torch.cuda.empty_cache()
    """) + model_section(
    "4.3", "Modèle 2 — Réseau dense MLP (from scratch)",
    """
Réseau de neurones « classique » du cours (ANN) : aucune convolution. L'image est réduite à 64 × 36, mise
à plat en un vecteur de 6 912 valeurs, puis traversée par deux couches cachées denses (512 puis 128
neurones, BatchNorm, ReLU, Dropout 0,3). Il ne connaît pas la notion de voisinage entre pixels : il sert de
point de comparaison pour mesurer ce qu'apportent les convolutions.

| Couche | Sortie |
|---|---|
| Réduction de l'image + mise à plat | 6 912 |
| Dense 1 (+ BatchNorm, ReLU, Dropout) | 512 |
| Dense 2 (+ BatchNorm, ReLU, Dropout) | 128 |
| Dense + sigmoïde | 3 |
""",
    [src_of("CorridorMLP")],
    """
    set_seed(cfg.seed)
    m = CorridorMLP()
    model_summary(m, (3, 144, 256), depth=1)
    """,
    "80 époques au plus, patience 15, AdamW 1e-3.",
    """
    fit_bo3('bo3_mlp', m, pretrained=False)
    del m; torch.cuda.empty_cache()
    """) + model_section(
    "4.4", "Modèle 3 — ResNet-18 pré-entraîné sur ImageNet",
    """
Transfert d'apprentissage comme pour BO1, avec un réseau plus léger : le **ResNet-18** (18 couches, 4
étages de 2 blocs résiduels). Ses couches ImageNet sont conservées et ajustées ; la dernière couche est
remplacée par Dropout 0,3 + une couche dense 512 → 3, suivie d'une sigmoïde.

| Partie | Origine | Sortie |
|---|---|---|
| conv1 + 4 étages résiduels | **ImageNet**, ajustés | 512 × 5 × 8 |
| Moyenne globale | — | 512 |
| Dropout + Dense 512 → 3 + sigmoïde | **nouvelle** | 3 |
""",
    [src_of("PretrainedRegressor")],
    """
    set_seed(cfg.seed)
    m = PretrainedRegressor()
    model_summary(m, (3, 144, 256), depth=2, pretrained=True)
    """,
    "80 époques au plus, patience 15 ; taux 1e-4 pour les couches ImageNet, 1e-3 pour la nouvelle tête.",
    """
    fit_bo3('bo3_resnet18', m, pretrained=True)
    del m; torch.cuda.empty_cache()
    """) + [
    md("## 5. Récapituler l'entraînement"),
    code("""
    rows = [{'BO': bo, 'modèle': f"{s['name']} ({s['init']})", 'id': s['id'],
             **{k: SUMMARY.get(s['id'], {}).get(k) for k in ['params_M', 'best_epoch', 'train_seconds', 'ms_per_item']}}
            for bo, lst in BO_MODELS.items() for s in lst]
    display(pd.DataFrame(rows).round(2))
    save_json({'notebook': '03', 'env': ENV, 'config': cfg.to_dict(), 'bo3_size': BO3_SIZE, 'models': SUMMARY},
              os.path.join(RES, '03_training.json'))
    """),
    md("""
## Observations

*Rédigées à partir des sorties de l'exécution, dans la copie exécutée de ce notebook
(`outputs/executed/03_modelisation_executed.ipynb`).*
"""),
]

write("03_modelisation.ipynb", nb03)
