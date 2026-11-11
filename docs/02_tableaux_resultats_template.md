# Gabarits de tableaux de résultats

**Toutes les valeurs sont à remplir après exécution réelle**, en recopiant les sorties des notebooks
(`outputs/results/*.json`, tableaux affichés dans 03 à 07). Ne pas arrondir de façon favorable, ne pas
retirer de ligne défavorable.

## Tableau comparatif général

| Modèle | BO | Pré-entraîné | Métrique | Valeur à remplir après exécution | Forces | Limites |
|---|---|---|---|---|---|---|
| MLP | BO1 | non | F1 (patch, test) | | | |
| LeNet-5 | BO1 | non | F1 (patch, test) | | | |
| AlexNet | BO1 | ImageNet | F1 (patch, test) | | | |
| VGG-16 | BO1 | ImageNet | F1 (patch, test) | | | |
| ResNet-50 | BO1 | ImageNet | F1 / AUROC (patch, test) | | | |
| ResNet-50 | BO1 | non | F1 / AUROC (patch, test) | | | |
| AE vanilla | BO1 | non | AUROC (patch, test) | | | |
| AE convolutionnel | BO1 | non | AUROC (patch, test) | | | |
| AE débruiteur | BO1 | non | AUROC (patch, test) | | | |
| DeepLabV3+ R101 zero-shot (MSP) | BO2 | Cityscapes | AP / FPR95 (pixel, test) | | | |
| DeepLabV3+ R101 fine-tuné | BO2 | Cityscapes | IoU / Dice (pixel, test) | | | |
| DeepLabV3+ MobileNetV2 + règles | BO3 | Cityscapes | accord de passabilité | | | |
| CNN + LSTM | ext. | ImageNet (gelé) | F1 (séquences simulées) | | | |
| CNN + GRU | ext. | ImageNet (gelé) | F1 (séquences simulées) | | | |

## BO1 — détail (test, seuil choisi sur validation)

| Modèle | Precision | Recall | F1 | AUROC | AP | FPR95 | Paramètres | Temps d'entraînement |
|---|---|---|---|---|---|---|---|---|
| … | à remplir | à remplir | à remplir | à remplir | à remplir | à remplir | à remplir | à remplir |

## BO2 — par famille (test)

| Famille | Nb d'images test | IoU moyen | Dice moyen | Rappel par objet |
|---|---|---|---|---|
| Faune | à remplir | à remplir | à remplir | à remplir |
| Débris / objets perdus | à remplir | à remplir | à remplir | à remplir |
| Infrastructure temporaire | à remplir | à remplir | à remplir | à remplir |
| Autre | à remplir | à remplir | à remplir | à remplir |

## BO2 — par taille d'objet (test)

| Taille | Nb d'objets | Rappel par objet (couverture ≥ 50 %) |
|---|---|---|
| petit (< 32² px) | à remplir | à remplir |
| moyen | à remplir | à remplir |
| grand (> 96² px) | à remplir | à remplir |

## BO3 — passabilité (test)

| Image | Famille | Occupation (%) | Classe | Libre gauche | Libre droite | Passabilité | Criticité | Validation humaine |
|---|---|---|---|---|---|---|---|---|
| … | | à remplir | à remplir | à remplir | à remplir | à remplir | à remplir | à remplir |

Écart moyen d'occupation GT vs prédiction (points) : *à remplir après exécution* ·
Accord de passabilité GT vs prédiction : *à remplir après exécution* ·
Accord avec le jugement humain : *à remplir après étiquetage*.
