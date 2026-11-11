# Plan de la présentation (slide par slide)

Aucune figure de résultat n'est préparée à l'avance : chaque visuel marqué **[après exécution]** est
une sortie réelle d'un notebook (`outputs/figures/`) ou une capture faite par l'étudiant.

| # | Titre | Contenu | Visuel |
|---|---|---|---|
| 1 | Titre | projet, module, équipe, date | — |
| 2 | Contexte | véhicules autonomes et objets inattendus | photo d'exemple du dataset (fichier réel) |
| 3 | Problématique | pas de classe « anomalie » connue, peu d'images annotées | schéma |
| 4 | Objectifs métier | BO1, BO2, BO3 en une phrase chacun | tableau BO |
| 5 | Objectifs data science | DSO par BO | tableau DSO |
| 6 | Dataset | structure, familles, audit | `01_samples_*.png` **[après exécution]** |
| 7 | Data understanding | déséquilibre, tailles d'objets, scènes dupliquées | `01_imbalance_sizes.png` **[après exécution]** |
| 8 | Méthodologie | pipeline, split groupé, protocole | schéma du pipeline |
| 9 | Modèles | familles du cours + 3 pré-entraînés imposés | tableau |
| 10 | **BO1** — détection | comparaison MLP / CNN / AE | `03_roc_pr.png` + tableau **[après exécution]** |
| 11 | **BO2** — localisation | zero-shot vs fine-tuning, IoU / Dice par famille | `04_best_cases.png`, `04_worst_cases.png` **[après exécution]** |
| 12 | **BO3** — emprise et passabilité | occupation, espace libre, message métier | `05_passability_*.png` **[après exécution]** |
| 13 | Extension RNN | séquences simulées, statut exploratoire | `06_rnn_roc_pr.png` **[après exécution]** |
| 14 | Démonstration web | fonctionnement de l'app, captures | captures **à réaliser par l'étudiant après exécution** |
| 15 | Limites | effectifs, domain gap, perspective, seuils hypothétiques | — |
| 16 | Perspectives | vidéo réelle, calibration, plus de données, OoD | — |
| 17 | Conclusion | réponse à chaque BO | chiffres clés **[après exécution]** |
