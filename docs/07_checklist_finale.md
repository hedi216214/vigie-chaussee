# Checklist finale

## Exécution (Google Colab, GPU T4)
- [ ] Copier `RoadAnomaly_BO_Project/` dans `MyDrive/`
- [ ] `00_main.ipynb` lancé en mode rapide (`QUICK = True`) sans erreur, puis en mode complet (`QUICK = False`)
- [ ] 01 : exploration et audit exécutés ; familles contrôlées visuellement ; `FAMILY_OVERRIDES` mis à jour si besoin
- [ ] 02 : split sauvegardé (`outputs/splits/splits.json`) — **ne plus le modifier ensuite**
- [ ] 03 : BO1 exécuté (MLP, LeNet-5, AlexNet, VGG-16, ResNet-50, autoencodeurs, option famille)
- [ ] 04 : BO2 exécuté ; vérifier que la source du modèle est bien « Cityscapes » (sinon signaler le repli)
- [ ] 05 : BO3 exécuté ; contrôle visuel de la chaussée ; gabarit de validation humaine rempli (recommandé)
- [ ] 06 : extension RNN exécutée et présentée comme exploratoire
- [ ] 07 : tableaux de synthèse générés
- [ ] Notebooks exécutés récupérés dans `outputs/executed/` (ce sont eux qui sont rendus, pas ceux de `outputs_quick/`)

## Documents
- [ ] Tous les « à remplir après exécution » remplacés par les valeurs réelles (ou supprimés avec justification)
- [ ] Observations rédigées dans chaque notebook (sections « à rédiger après exécution »)
- [ ] Rapport rédigé selon `docs/03_plan_rapport_technique.md`, chaque chiffre relié à sa source
- [ ] Seuils BO3 et poids de criticité justifiés (et sensibilité montrée)
- [ ] Limites et biais discutés (`docs/06_analyse_critique.md` comme base)
- [ ] Références vérifiées (article ICCV 2019, dépôt VainF, autres jeux de données cités)

## Application et démo
- [ ] Plateforme Vigie Chaussée construite à partir de `outputs/deploy.zip` (notebook 05)
- [ ] Plateforme ouverte dans le navigateur et testée sur des images de test
- [ ] Captures d'écran réelles réalisées pour la slide « démonstration web »
- [ ] Vidéo filmée selon `docs/05_scenario_demo.md` (optionnel)

## Présentation
- [ ] Slides selon `docs/04_plan_presentation.md`, une slide par BO avec figures réelles
- [ ] Slide limites + perspectives

## Rendu
- [ ] Archive : notebooks exécutés, `src/`, `platform/`, rapport, slides, (vidéo), README
- [ ] Pas de données personnelles ni de jetons d'accès dans les fichiers rendus
