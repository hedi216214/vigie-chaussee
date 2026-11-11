# Plan du rapport technique

Règle : chaque chiffre et chaque figure du rapport provient d'une sortie réelle des notebooks
(indiquer le notebook et le fichier source sous chaque tableau / figure).

1. **Introduction**
   - Contexte : perception des véhicules autonomes face aux objets rares (hors classes connues).
   - Problématique : détecter, localiser et qualifier des obstacles inattendus avec très peu d'images annotées.
   - Contributions du projet (une phrase par BO).
2. **Dataset RoadAnomaly**
   - Source et référence (Lis et al., ICCV 2019), structure des fichiers, format des masques (0 / 2).
   - Inventaire mesuré (notebook 01) : nombre d'images, résolution, familles, scènes multi-images.
   - Audit qualité : cohérence image / masque, composantes minuscules, masques d'instances manquants.
   - Limites : petit volume, photos du web, pas de masque de chaussée, pas de calibration, pas de vidéo.
3. **Méthodologie**
   - BO / DSO (tableau de correspondance), taxonomie métier et sa règle de construction.
   - Prétraitement et split groupé (justification anti-fuite).
   - Modèles par famille du cours (ANN, CNN, autoencodeurs, RNN) et modèles pré-entraînés imposés.
   - BO3 : reconstruction de la chaussée visible, définitions des grandeurs, règles de passabilité, score de criticité.
4. **Expériences**
   - Hyperparamètres, optimiseurs, régularisation, augmentation (renvoi à `docs/01_protocole_experimental.md`).
   - Sélection sur validation, environnement (GPU, versions).
5. **Résultats** (tableaux et figures à remplir après exécution)
   - BO1 : tableau comparatif, ROC / PR, matrices de confusion, reconstructions d'autoencodeurs.
   - BO2 : zero-shot vs fine-tuning, IoU / Dice par famille et par taille, rappel par objet, cas réussis / échecs.
   - BO3 : chaussée prédite (contrôle qualité), occupation, passabilité, sensibilité aux seuils, messages.
   - Extension RNN : résultats exploratoires, clairement séparés.
6. **Discussion critique**
   - Forces, faiblesses, biais (dataset, familles, cadrage), domain gap Cityscapes.
   - Incertitude statistique (10 images de test, intervalles bootstrap).
   - Généralisation (catégories jamais vues, conditions absentes : nuit, pluie).
   - Validité des hypothèses BO3 (perspective, bords de route, seuils).
7. **Conclusion et perspectives**
   - Réponse à chaque BO, modèle recommandé, prochaines étapes (vidéo réelle, calibration, plus de données).
8. **Annexes** : versions logicielles, commandes de reproduction, structure du code, lien vers l'application.
