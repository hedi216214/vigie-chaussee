# Analyse critique méthodologique (sans résultat chiffré)

## 1. Biais et limites du dataset
- **Volume très faible** : toutes les conclusions reposent sur 10 images de test ; une seule image peut
  déplacer fortement une moyenne. → intervalles bootstrap, prudence sur les écarts faibles.
- **Familles déséquilibrées** et déduites du nom de fichier (pas d'annotation officielle de famille).
  → effectifs par famille rapportés avec chaque métrique ; certaines familles sont anecdotiques en test.
- **Photos du web** : cadrages, hauteurs de caméra et objectifs variés, éloignés d'une caméra embarquée.
- **Scènes quasi dupliquées** : gérées par le split groupé ; sans cela, le score serait surestimé.
- **Annotation** : petites composantes isolées (bruit possible), choix d'annotation discutables (objet
  partiellement annoté, frontière anomalie / chaussée), masques d'instances incomplets.
- **Conditions absentes** : nuit, pluie, brouillard, vidéo → généralisation non évaluable ici.

## 2. Limites par BO
- **BO1** : la classification de patches est un **proxy** de la présence dans une scène ; les patches
  ambigus (1–5 % d'anomalie) sont exclus, ce qui facilite artificiellement la tâche. Les autoencodeurs
  peuvent reconstruire correctement des objets nouveaux (pas de détection) ou mal reconstruire des textures
  normales complexes (fausses alarmes).
- **BO2** : domain gap Cityscapes → RoadAnomaly ; en zero-shot, l'incertitude se concentre aussi sur les
  contours d'objets et les classes peu vues (fausses alarmes structurelles). Le fine-tuning sur 40 images
  risque le sur-apprentissage (early stopping, augmentation, LR faible sur l'encodeur). F1 et Dice pixel
  sont identiques par construction ; l'IoU global favorise les gros objets → rappel par objet indispensable.
- **BO3** : sans calibration, le **taux d'occupation dépend de la perspective** (objet lointain = peu de
  pixels). La chaussée cachée par l'obstacle est reconstruite par interpolation des bords : hypothèse fragile
  en virage, intersection, route non goudronnée ou entièrement couverte (troupeau). Les seuils de
  passabilité et les poids de criticité sont des **hypothèses** ; leur sensibilité doit être montrée. Les
  erreurs BO2 se propagent à BO3 (d'où la double évaluation GT / prédiction).
- **RNN** : séquences simulées par zoom 2D : ni parallaxe, ni mouvement propre de l'obstacle ; résultats non
  transposables à une vraie vidéo.

## 3. Erreurs attendues (à confirmer ou infirmer avec les figures réelles)
- Petits débris lointains manqués (faible nombre de pixels, sous-échantillonnage à 512×288).
- Animaux partiellement occultés ou de couleur proche du décor : segmentation incomplète.
- Groupes de cônes : certains éléments manqués, rappel par objet inférieur au rappel pixel.
- Fausses alarmes : piétons, véhicules, panneaux, ombres fortes, bas-côtés (objets « inhabituels » mais
  non anormaux au sens du dataset).
- Chaussée non détectée sur routes de terre ou enneigées → passabilité « indéterminée ».

## 4. Pistes d'amélioration
- Données : plus d'images (autres jeux d'anomalies routières, à vérifier : Lost and Found, Fishyscapes,
  SegmentMeIfYouCan), annotation de la chaussée sur un sous-ensemble pour valider BO3.
- Modèles : résolution plus élevée pour les petits objets, pertes focalisées sur les objets (Lovász, focal),
  méthodes OoD dédiées (resynthèse d'image comme dans l'article de référence, outlier exposure).
- BO3 : estimation de l'horizon / homographie approximative pour corriger la perspective, segmentation de
  voies (marquages), validation humaine systématique.
- Évaluation : validation croisée groupée (k plis) pour réduire la variance due au petit test.
- Temporel : séquences vidéo réelles pour une vraie évaluation LSTM / GRU.
