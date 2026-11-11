# Protocole expérimental

Toutes les valeurs ci-dessous sont des **choix a priori** (point de départ raisonnable), à justifier ou
ajuster **sur la validation**. Les valeurs réellement utilisées sont enregistrées dans `outputs/results/*.json`.

## 1. Données

| Élément | Choix | Justification |
|---|---|---|
| Split | 40 / 10 / 10 images, stratifié par famille, **groupé par scène** | les séries d'une même scène (rochers, pneus, hérissons) restent en train : pas de fuite de scène |
| Graine | 42 (Python, NumPy, PyTorch) | reproductibilité |
| Résolution | 512 × 288 (16:9) | multiples de 32, tient en mémoire T4 avec batch 4 |
| Normalisation | moyenne / écart-type ImageNet | convention des encodeurs pré-entraînés (ImageNet, Cityscapes VainF) |
| Masques | redimensionnement au plus proche voisin | évite des valeurs de classe interpolées |
| Patches BO1 | 96 × 96, pas 48, positif ≥ 5 % de pixels anormaux, négatif = 0 % | crée des négatifs (toutes les images contiennent une anomalie) |
| Séquences RNN | 8 étapes simulées (zoom progressif), 6 observées | exploration uniquement |

## 2. Augmentation (train uniquement)

Flip horizontal (p = 0,5) ; recadrage-zoom aléatoire (60–100 % de l'image) ; luminosité, contraste,
saturation ± 25 %. Pas de flip vertical (la route reste en bas). Même transformation géométrique pour
l'image et le masque.

## 3. Hyperparamètres par modèle

| Modèle | Entrée | Optimiseur | LR (pré-entraîné / tête) | Époques max | Patience | Batch | Perte | Régularisation |
|---|---|---|---|---|---|---|---|---|
| MLP | 32×32 | AdamW | — / 1e-3 | 40 | 8 | 64 | CE pondérée | dropout 0,3, BatchNorm, weight decay 1e-4 |
| LeNet-5 | 32×32 | AdamW | — / 1e-3 | 40 | 8 | 64 | CE pondérée | weight decay, early stopping |
| AlexNet (ImageNet) | 224 | AdamW | 1e-4 / 1e-3 | 20 | 8 | 64 | CE pondérée | dropout d'origine, weight decay |
| VGG-16 (ImageNet) | 224 | AdamW | 1e-4 / 1e-3 | 20 | 8 | 32 | CE pondérée | dropout d'origine, weight decay |
| ResNet-50 (ImageNet) | 224 | AdamW | 1e-4 / 1e-3 | 20 | 8 | 32 | CE pondérée | weight decay, augmentation |
| ResNet-50 (de zéro) | 224 | AdamW | — / 1e-3 | 20 | 8 | 32 | CE pondérée | idem (même budget) |
| AE vanilla / conv | 96×96 | AdamW | — / 1e-3 | 40 | 8 | 64 | MSE | goulot d'étranglement |
| AE débruiteur | 96×96 | AdamW | — / 1e-3 | 40 | 8 | 64 | MSE (entrée + bruit σ = 0,1) | bruit d'entrée |
| Famille (BO1-b) | 224 | AdamW | gelé / 1e-3 | 40 | meilleur val | 16 | CE | backbone gelé |
| DeepLabV3+ R101 (BO2) | 512×288 | AdamW + cosinus | 1e-4 / 1e-3 | 60 | 15 | 4 | BCE (pos_weight ≤ 10) + Dice | weight decay, augmentation, AMP |
| DeepLabV3+ MobileNetV2 (BO3) | 1024×576 | — (pré-entraîné, non ré-entraîné) | — | — | — | 1 | — | — |
| CNN + LSTM / GRU | 6 × 224×128 | AdamW | gelé / 1e-3 | 60 | meilleur val | 32 | BCE | encodeur gelé |

## 4. Sélection et évaluation

1. Meilleure époque choisie sur la validation (F1 pour BO1, Dice pour BO2).
2. Seuil de décision choisi sur la validation (F1 / Dice maximal), appliqué tel quel au test.
3. Le test n'est utilisé qu'une fois, à la fin.
4. Métriques : precision, recall, F1, IoU, Dice ; AUROC, AP (AuPRC), FPR95 sans seuil ; par famille,
   par taille d'objet (COCO-like : < 32², 32²–96², > 96² pixels à 512×288), rappel par objet.
5. Incertitude : intervalle de confiance bootstrap de l'IoU moyen par image (10 images de test).

## 5. Seuils métier BO3 (hypothèses)

| Paramètre | Valeur par défaut | Rôle |
|---|---|---|
| Classes d'occupation | < 5 %, 5–15 %, 15–30 %, > 30 % | cahier des charges |
| `side_ok` | 30 % de la largeur de chaussée | espace libre minimal d'un côté pour passer |
| `side_comfort` | 45 % | « marge suffisante » au-delà, « réduite » en deçà |
| `side_tight` | 15 % | en dessous partout : voie bloquée |
| `blocked_lateral` | 90 % | emprise latérale bloquante |
| Poids de criticité | occupation 0,35 · passabilité 0,35 · proximité 0,20 · famille 0,10 | heuristique |

Analyse de sensibilité prévue (notebook 05) : `side_ok` ∈ {0,20 ; 0,30 ; 0,40}.
