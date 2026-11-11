from common import INTEGRITY, code, md, setup_cells, write

# ======================================================================================
# 01 — Exploration et audit (CRISP-DM : compréhension des données)
# ======================================================================================
nb01 = [md("""
# 01 — Exploration du jeu de données RoadAnomaly

RoadAnomaly (Lis et al., ICCV 2019) réunit 60 photos de route prises depuis un véhicule, chacune avec un
objet inattendu sur la chaussée (animal, cône, objet perdu, engin…) et un masque qui indique, pixel par
pixel, où se trouve cet objet. Avant de modéliser, on vérifie ce que contient vraiment le jeu de données.
"""), md("""
### Étapes de ce notebook

1. Importer les bibliothèques
2. Charger la liste des images et leur catégorie
3. Visualiser des images et leurs masques
4. Auditer les annotations (qualité des masques)
5. Mesurer le déséquilibre obstacle / fond et la taille des objets
6. Repérer les familles sous-représentées
7. Définir les scénarios de sécurité
8. Sauvegarder le résumé
"""), INTEGRITY] + setup_cells() + [
    md("## 1. Importer les bibliothèques"),
    code("""
    import numpy as np, pandas as pd, matplotlib.pyplot as plt
    from src.dataset import (FAMILIES, family_of, group_of, load_arrays, audit_annotations,
                             object_sizes, size_bin)
    from src.visualize import show_samples, overlay
    """),
    md("""
## 2. Charger la liste des images et leur catégorie

Le jeu de données ne fournit pas de catégorie : on la déduit du préfixe du nom de fichier
(`animals…`, `cones…`, `obstacles…`, `lost…`, `vehicle…`) et on regroupe en quatre familles métier :
**faune**, **débris / objets perdus**, **infrastructure temporaire** (cônes, barrières) et **autre**
(véhicules inhabituels). Certaines images sont des vues successives d'une même scène (quasi-doublons) :
on les repère pour ne pas les répartir entre entraînement et test.
"""),
    code("""
    names = list_frames(ROOT)
    inv = pd.DataFrame({'name': names, 'family': [family_of(n) for n in names],
                        'group': [group_of(n) for n in names]})
    counts = inv.family.value_counts().reindex(FAMILIES, fill_value=0)
    print(len(names), 'images listées dans frame_list.json')
    display(counts.to_frame('n_images').assign(part=lambda d: (d.n_images / d.n_images.sum()).round(3)))
    multi = inv.group.value_counts()
    print('Scènes à plusieurs images (quasi-doublons) :')
    display(multi[multi > 1])
    """),
    md("""
## 3. Visualiser des images et leurs masques

Trois exemples par famille ; l'obstacle annoté est superposé en couleur. Ce contrôle visuel permet de
vérifier que les familles déduites des noms de fichiers sont cohérentes.
"""),
    code("""
    for fam in FAMILIES:
        sel = inv[inv.family == fam].name.tolist()[:3]
        if not sel:
            continue
        ims, ms = load_arrays(ROOT, sel, (W, H))
        show_samples(ims, ms, sel, n=len(sel), cols=3, path=os.path.join(FIG, f'01_samples_{fam}.png'))
        plt.suptitle(fam); plt.show()
    """),
    md("""
## 4. Auditer les annotations

Contrôles automatiques sur chaque couple image / masque : image lisible, même taille que le masque,
valeurs attendues dans le masque (0 = fond, 2 = obstacle), présence d'au moins un pixel d'obstacle,
composantes minuscules (souvent du bruit d'annotation) et présence d'un masque d'instances.
"""),
    code("""
    audit = audit_annotations(ROOT, names)
    audit.to_csv(os.path.join(RES, '01_audit_annotations.csv'), index=False)
    flag = lambda c: audit[c].fillna(False).astype(bool)
    checks = {
        'images_illisibles': int(audit.error.notna().sum()),
        'taille_image_differente_du_masque': int((~flag('size_match')).sum()),
        'valeurs_inattendues_dans_masque': int(flag('unexpected_values').sum()),
        'masques_sans_anomalie': int((audit.anomaly_fraction == 0).sum()),
        'images_avec_composantes_minuscules': int((audit.n_tiny_components > 0).sum()),
        'images_sans_masque_instance': int((~flag('has_instance_mask')).sum()),
    }
    display(pd.Series(checks, name='nombre d\\'images'))
    display(audit.groupby(['img_w', 'img_h']).size().rename('n_images'))
    display(audit[['mask_values']].value_counts().rename('n_images'))
    """),
    md("Les images qui contiennent des composantes minuscules sont affichées pour un contrôle visuel :"),
    code("""
    tiny = audit[audit.n_tiny_components > 0].sort_values('n_tiny_components', ascending=False).name.tolist()[:6]
    if tiny:
        native = (int(audit.img_w.mode()[0]), int(audit.img_h.mode()[0]))   # résolution la plus fréquente
        ims, ms = load_arrays(ROOT, tiny, native)
        show_samples(ims, ms, tiny, n=len(tiny), path=os.path.join(FIG, '01_tiny_components.png')); plt.show()
    else:
        print('Aucune composante minuscule détectée.')
    """),
    md("""
## 5. Mesurer le déséquilibre et la taille des objets

Part des pixels d'obstacle par image (le fond domine largement) et surface de chaque objet à la
résolution de travail 512 × 288. Ces deux mesures guident les choix de la suite : perte pondérée,
métriques au pixel et par objet.
"""),
    code("""
    print(audit.anomaly_fraction.describe())
    display(audit.groupby('family').anomaly_fraction.agg(['count', 'mean', 'median', 'min', 'max']))

    ims, ms = load_arrays(ROOT, names, (W, H))
    objs = object_sizes(ms, names)
    objs['size'] = objs.area_px.map(size_bin)
    objs.to_csv(os.path.join(RES, '01_objects.csv'), index=False)
    display(pd.crosstab(objs.family, objs['size'], margins=True))

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    audit.anomaly_fraction.plot.hist(bins=20, ax=ax[0], title="part de pixels d'obstacle par image")
    np.log10(objs.area_px).plot.hist(bins=30, ax=ax[1], title=f'log10(surface des objets, px) à {W}×{H}')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, '01_imbalance_sizes.png'), dpi=150); plt.show()
    """),
    md("""
## 6. Repérer les familles sous-représentées

Une famille qui représente moins de 15 % des images est jugée sous-représentée : ses résultats seront
peu fiables. Le fichier `01_criticite_template.csv` permet de noter la gravité de chaque famille.
"""),
    code("""
    UNDER_REP = 0.15
    share = counts / counts.sum()
    under = share[share < UNDER_REP]
    print('Familles sous-représentées :', list(under.index))
    crit = pd.DataFrame({'famille': FAMILIES, 'frequence_dataset': share.reindex(FAMILIES).values,
                         'criticite_expert_1a3': [None] * len(FAMILIES),
                         'justification': [''] * len(FAMILIES)})
    crit.to_csv(os.path.join(RES, '01_criticite_template.csv'), index=False)
    crit
    """),
    md("""
## 7. Définir les scénarios de sécurité

| Scénario | Familles | Risque principal | BO concernés |
|---|---|---|---|
| S1 Animal sur la voie du véhicule | faune | mouvement imprévisible, collision | BO1, BO2, BO3 |
| S2 Petit débris lointain | débris | détection tardive (objet de quelques pixels) | BO1, BO2 |
| S3 Cônes ou barrières qui réduisent la voie | infrastructure | mauvaise estimation de l'espace libre | BO2, BO3 |
| S4 Troupeau ou obstacle qui bloque la chaussée | faune, débris | voie bloquée, arrêt nécessaire | BO3 |

Le fichier `01_scenarios_template.csv` permet d'associer chaque image à un scénario.
"""),
    code("""
    scen = inv[['name', 'family']].assign(scenario='')
    scen.to_csv(os.path.join(RES, '01_scenarios_template.csv'), index=False)
    """),
    md("## 8. Sauvegarder le résumé"),
    code("""
    save_json({'notebook': '01', 'env': ENV, 'n_images': len(names),
               'images_par_famille': counts.to_dict(), 'controles_audit': checks,
               'objets_par_taille': objs['size'].value_counts().to_dict(),
               'familles_sous_representees': list(under.index)},
              os.path.join(RES, '01_exploration.json'))
    """),
    md("""
## Conclusion

*Rédigée à partir des sorties de l'exécution, dans la copie exécutée de ce notebook
(`outputs/executed/01_exploration_dataset_executed.ipynb`).*
"""),
]

# ======================================================================================
# 02 — Préparation des données (CRISP-DM : préparation des données)
# ======================================================================================
nb02 = [md("""
# 02 — Préparation des données : découpage, mise à l'échelle, patches

Ce notebook prépare une fois pour toutes les données utilisées par les trois objectifs. Toutes les
décisions prises ici sont réutilisées telles quelles par les notebooks 03 à 05.
"""), md("""
### Étapes de ce notebook

1. Importer les bibliothèques
2. Découper en entraînement / validation / test (40 / 10 / 10)
3. Mettre les images à la résolution de travail (512 × 288) et les mettre en cache
4. Calculer le poids des pixels d'obstacle (déséquilibre)
5. Vérifier l'augmentation de données
6. Découper les images en zones de 96 × 96 pour BO1
7. Sauvegarder les choix de préparation
"""), INTEGRITY] + setup_cells() + [
    md("## 1. Importer les bibliothèques"),
    code("""
    import numpy as np, pandas as pd, matplotlib.pyplot as plt
    from src.dataset import make_splits, check_no_leakage, load_arrays, family_of, augment_pair, extract_patches
    from src.visualize import overlay
    """),
    md("""
## 2. Découper en entraînement / validation / test

- **Entraînement (40 images)** : les modèles apprennent dessus.
- **Validation (10 images)** : sert à choisir le meilleur état de chaque modèle, son seuil de décision et
  le meilleur modèle de chaque BO.
- **Test (10 images)** : n'est regardé qu'à la fin, pour mesurer les performances.

Le découpage est stratifié par catégorie (chaque catégorie est représentée partout) et groupé par scène :
les images d'une même scène restent toutes en entraînement, pour qu'aucune scène du test n'ait été vue.
"""),
    code("""
    names = list_frames(ROOT)
    splits = make_splits(names, n_val=10, n_test=10, seed=cfg.seed)
    check_no_leakage(splits)
    save_json(splits, os.path.join(paths.sub('splits'), 'splits.json'))
    rows = [(s, family_of(n)) for s, lst in splits.items() for n in lst]
    display(pd.crosstab(pd.Series([f for _, f in rows], name='famille'),
                        pd.Series([s for s, _ in rows], name='split'), margins=True))
    """),
    md("""
## 3. Mettre les images à la résolution de travail

Toutes les images et tous les masques sont ramenés à 512 × 288 (format 16:9, dimensions multiples de 32
comme l'exigent les encodeurs convolutifs) et enregistrés dans un fichier unique, relu par les notebooks
suivants.
"""),
    code("""
    arrays = {s: load_arrays(ROOT, lst, (W, H)) for s, lst in splits.items()}
    np.savez_compressed(os.path.join(paths.sub('processed'), f'arrays_{W}x{H}.npz'),
                        **{f'{s}_images': a[0] for s, a in arrays.items()},
                        **{f'{s}_masks': a[1] for s, a in arrays.items()},
                        **{f'{s}_names': np.array(splits[s]) for s in arrays})
    for s, (im, m) in arrays.items():
        print(f'{s:5s} images {im.shape}  pixels d\\'obstacle {m.mean():.4f}')
    """),
    md("""
## 4. Calculer le poids des pixels d'obstacle

Les obstacles couvrent une petite part des pixels. Sans correction, un modèle qui répond « fond »
partout aurait l'air bon. On donne donc plus de poids aux pixels d'obstacle dans la perte de BO2 :
poids = (part de fond) / (part d'obstacle), calculé sur l'entraînement seulement et plafonné à 10.
"""),
    code("""
    pos = float(arrays['train'][1].mean())
    pos_weight = min((1 - pos) / pos, cfg.max_pos_weight)
    print(f'part de pixels d\\'obstacle (entraînement) = {pos:.4f} -> poids = {pos_weight:.2f} '
          f'(plafonné à {cfg.max_pos_weight})')
    """),
    md("""
## 5. Vérifier l'augmentation de données

Avec 40 images, on crée des variantes à chaque passage : miroir horizontal, zoom / recadrage, variation
de luminosité, de contraste et de couleur. Le masque subit la même transformation géométrique que l'image.
"""),
    code("""
    rng = np.random.default_rng(cfg.seed)
    img, m = arrays['train'][0][0], arrays['train'][1][0]
    fig, axes = plt.subplots(2, 3, figsize=(13, 5))
    for ax in axes.ravel():
        a, b = augment_pair(img, m, rng)
        ax.imshow(overlay(a, b)); ax.axis('off')
    fig.suptitle('Six augmentations de la même image (masque transformé avec elle)')
    fig.savefig(os.path.join(FIG, '02_augmentations.png'), dpi=120); plt.show()
    """),
    md("""
## 6. Découper les images en zones de 96 × 96 pour BO1

Toutes les images contiennent un obstacle : au niveau de l'image entière, il n'y aurait aucun exemple
« sans obstacle ». On découpe donc chaque image en zones de 96 × 96 pixels (pas de 48) :
- zone **avec obstacle** si au moins 5 % de ses pixels sont des pixels d'obstacle ;
- zone **sans obstacle** si elle n'en contient aucun ;
- les zones entre les deux (bords d'objets) sont écartées.
"""),
    code("""
    PATCH, STRIDE, POS_MIN = 96, 48, 0.05
    patch_data, patch_meta = {}, {}
    for s, (im, m) in arrays.items():
        X, y, meta = extract_patches(im, m, splits[s], patch=PATCH, stride=STRIDE, pos_min=POS_MIN)
        patch_data[f'X_{s}'], patch_data[f'y_{s}'] = X, y
        patch_meta[s] = meta
        print(f'{s:5s} zones {len(y):5d}  avec obstacle {int(y.sum()):4d}  sans obstacle {int((y == 0).sum()):4d}')
    np.savez_compressed(os.path.join(paths.sub('processed'), f'patches_{PATCH}.npz'), **patch_data)
    save_json(patch_meta, os.path.join(paths.sub('processed'), f'patches_{PATCH}_meta.json'))
    """),
    md("## 7. Sauvegarder les choix de préparation"),
    code("""
    save_json({'notebook': '02', 'env': ENV, 'config': cfg.to_dict(), 'split_sizes': {k: len(v) for k, v in splits.items()},
               'pixel_pos_fraction_train': pos, 'pos_weight': pos_weight,
               'patches': {'size': PATCH, 'stride': STRIDE, 'pos_min': POS_MIN,
                           **{s: {'n': int(len(patch_data[f'y_{s}'])), 'pos': int(patch_data[f'y_{s}'].sum())}
                              for s in arrays}}},
              os.path.join(RES, '02_preprocessing.json'))
    """),
    md("""
## Conclusion

*Rédigée à partir des sorties de l'exécution, dans la copie exécutée de ce notebook
(`outputs/executed/02_preprocessing_splits_executed.ipynb`).*
"""),
]

write("01_exploration_dataset.ipynb", nb01)
write("02_preprocessing_splits.ipynb", nb02)
