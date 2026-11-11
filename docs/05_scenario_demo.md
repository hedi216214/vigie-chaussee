# Scénario de la vidéo de démonstration (à filmer par l'étudiant)

Durée visée : 2 à 3 minutes. Tout ce qui apparaît à l'écran doit être une exécution réelle de
l'application ; ne pas monter de résultats, ne pas retoucher les valeurs affichées.

## Préparation
- Plateforme `platform/` construite avec `deploy.zip` du notebook 05 (fichier `model_int8.onnx`).
- Choisir 3 images **du jeu de test** (jamais vues à l'entraînement) : une avec un animal, une avec des
  débris, une avec des cônes. Option : une photo personnelle de route (montrer aussi les limites).
- Enregistreur d'écran (OBS, Xbox Game Bar, QuickTime), résolution 1080p.

## Déroulé

| Temps | Écran | Commentaire (à dire) |
|---|---|---|
| 0:00–0:15 | page d'accueil, bloc « État des modèles » | présentation du but et des modèles chargés |
| 0:15–0:45 | upload image 1 → Analyser | « le masque jaune est la prédiction du modèle BO2 » |
| 0:45–1:10 | overlay + carte de score | expliquer chaussée (bleu), anomalie sur chaussée (jaune) |
| 1:10–1:35 | rapport : occupation, passabilité, criticité, message | lire le message tel qu'affiché |
| 1:35–2:00 | déplacer le seuil | montrer le compromis fausses alarmes / oubli |
| 2:00–2:25 | image 2 puis image 3 | comparer les verdicts de passabilité |
| 2:25–2:45 | cas difficile ou échec | assumer une limite (petit objet, route non détectée…) |
| 2:45–3:00 | conclusion | rappeler : pourcentages de pixels, pas de calibration, démonstrateur académique |

## Script court (voix off)
« Cette application charge le meilleur modèle de segmentation entraîné sur RoadAnomaly. Je dépose une
image de test… Le modèle localise l'obstacle en jaune. La chaussée estimée apparaît en bleu ; l'application
calcule la part de la chaussée visible occupée par l'obstacle et l'espace libre de chaque côté, puis en
déduit une recommandation. Ces valeurs sont des rapports de pixels, sans calibration caméra : c'est un
démonstrateur, pas un système de conduite. »
