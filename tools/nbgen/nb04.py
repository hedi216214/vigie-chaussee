from common import INTEGRITY, code, load_split_cells, md, setup_cells, write

nb04 = [md("""
# 04 — Évaluation : comparer les 3 modèles de chaque objectif et choisir le meilleur

Règles communes, pour une comparaison équitable :
1. Le **seuil de décision** de chaque modèle (BO1, BO2) est choisi sur la **validation**.
2. Le **meilleur modèle de chaque BO est choisi sur la validation** (F1 pour BO1, Dice pour BO2,
   erreur absolue moyenne pour BO3), puis ses scores de **test** sont rapportés : le test ne sert jamais à choisir.
3. Avec 10 images de test, les écarts sont accompagnés d'intervalles de confiance bootstrap.

| BO | Métrique de choix (validation) | Métriques rapportées (test) |
|---|---|---|
| BO1 | F1 | precision, recall, F1, AUROC, AP, FPR95, matrice de confusion |
| BO2 | Dice | IoU, Dice = F1, precision, recall, AUROC, AP, FPR95, par catégorie, par taille d'objet |
| BO3 | erreur absolue moyenne | erreur sur l'occupation et l'espace libre (points), accord des classes d'occupation et de passabilité |
"""), md("""
### Étapes de ce notebook

1. Importer les bibliothèques et charger les prédictions du notebook 03
2. BO1 — comparer les 3 modèles de détection
3. BO2 — comparer les 3 modèles de localisation (global, par catégorie, par taille, cas extrêmes)
4. BO3 — comparer les 3 modèles du couloir (et une référence calculée sur la zone BO2)
5. Résultats : tableau de synthèse et sauvegarde
"""), INTEGRITY] + setup_cells() + load_split_cells() + [
    md("## 1. Importer les bibliothèques et charger les prédictions"),
    code("""
    import pandas as pd, matplotlib.pyplot as plt
    from PIL import Image
    from src.models_catalog import BO_MODELS
    from src.dataset import category_of
    from src.evaluate import (binary_classification_metrics, best_f1_threshold, segmentation_metrics, best_seg_threshold,
                              threshold_free_metrics, per_image_metrics, object_recall, bootstrap_ci)
    from src.corridor import corridor_targets, corridor_mask, regression_report
    from src.visualize import plot_confusion, plot_roc_pr, show_predictions

    TRAIN = load_json(os.path.join(RES, '03_training.json'))['models']
    label = {s['id']: f"{s['name']} ({s['init']})" for lst in BO_MODELS.values() for s in lst}
    BEST, EVAL = {}, {}
    def load_pred(mid): return np.load(os.path.join(PREDS, f'{mid}.npz'))
    """),
    md("""
## 2. BO1 — Détection de la présence d'un obstacle (zones 96 × 96)

Pour chaque modèle : seuil choisi sur la validation (meilleur F1), puis précision, rappel, F1, aire sous
la courbe ROC (AUROC), précision moyenne (AP) et taux de fausses alertes à 95 % de rappel (FPR95) sur le test.
"""),
    code("""
    rows, curves = [], {}
    for s in BO_MODELS['BO1']:
        z = load_pred(s['id'])
        thr, f1_val = best_f1_threshold(z['y_val'], z['s_val'])
        m = binary_classification_metrics(z['y_test'], z['s_test'], thr)
        rows.append({'id': s['id'], 'modèle': label[s['id']], 'F1 val': f1_val, 'seuil': thr,
                     **{k: m[k] for k in ['precision', 'recall', 'f1', 'auroc', 'ap', 'fpr95']},
                     'ms / patch': TRAIN[s['id']]['ms_per_item'], 'cm': m['confusion_matrix']})
        curves[label[s['id']]] = (z['y_test'], z['s_test'])
    bo1 = pd.DataFrame(rows).set_index('id')
    BEST['BO1'] = bo1['F1 val'].idxmax()
    display(bo1.drop(columns='cm').round(3))
    print('Meilleur modèle BO1 (choisi sur la validation) :', label[BEST['BO1']])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (mid, r) in zip(axes, bo1.iterrows()):
        cm = np.array(r['cm']); ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2): ax.text(j, i, cm[i, j], ha='center', va='center')
        ax.set_xticks([0, 1], ['sans', 'avec']); ax.set_yticks([0, 1], ['sans', 'avec'])
        ax.set_title(label[mid][:34], fontsize=9); ax.set_xlabel('prédit'); ax.set_ylabel('réel')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, '04_bo1_confusion.png'), dpi=130); plt.show()
    plot_roc_pr(curves, path=os.path.join(FIG, '04_bo1_roc_pr.png')); plt.show()
    EVAL['BO1'] = bo1.drop(columns='cm').reset_index().to_dict('records')
    """),
    md("""
## 3. BO2 — Localisation au pixel

IoU (intersection sur union) et Dice mesurent le recouvrement entre la zone prédite et l'obstacle réel.
L'intervalle de confiance à 95 % de l'IoU est obtenu par bootstrap sur les 10 images de test.
"""),
    code("""
    gt_val, gt_test, names_test = DATA['val'][1], DATA['test'][1], DATA['test'][2]
    rows, PROB, curves = [], {}, {}
    for s in BO_MODELS['BO2']:
        z = load_pred(s['id']); pv, pt = z['val'].astype(np.float32), z['test'].astype(np.float32)
        thr, dice_val = best_seg_threshold(pv, gt_val)
        m = segmentation_metrics(pt, gt_test, thr); free = threshold_free_metrics(pt, gt_test)
        pi = per_image_metrics(pt, gt_test, names_test, thr)
        conf = np.array([[((pt[i] >= thr) & (gt_test[i] > 0)).sum(), ((pt[i] >= thr) & (gt_test[i] == 0)).sum(),
                          ((pt[i] < thr) & (gt_test[i] > 0)).sum()] for i in range(len(gt_test))])
        rng = np.random.default_rng(cfg.seed); idx = rng.integers(0, len(gt_test), (2000, len(gt_test)))
        b = conf[idx].sum(1); ious = b[:, 0] / (b[:, 0] + b[:, 1] + b[:, 2])
        rows.append({'id': s['id'], 'modèle': label[s['id']], 'Dice val': dice_val, 'seuil': thr,
                     **{k: m[k] for k in ['iou', 'dice', 'precision', 'recall']}, **free,
                     'IoU IC95 bas': float(np.percentile(ious, 2.5)), 'IoU IC95 haut': float(np.percentile(ious, 97.5)),
                     'ms / image': TRAIN[s['id']]['ms_per_item']})
        PROB[s['id']] = (pt, thr, pi)
        curves[label[s['id']]] = (gt_test, pt)
    bo2 = pd.DataFrame(rows).set_index('id')
    BEST['BO2'] = bo2['Dice val'].idxmax()
    display(bo2.round(3))
    print('Meilleur modèle BO2 (choisi sur la validation) :', label[BEST['BO2']])
    plot_roc_pr(curves, path=os.path.join(FIG, '04_bo2_roc_pr.png')); plt.show()
    EVAL['BO2'] = bo2.reset_index().to_dict('records')
    """),
    md("**Par catégorie** (IoU moyen par image) **et par taille d'objet** (part des objets dont au moins la moitié est signalée) :"),
    code("""
    # Par catégorie (IoU moyen par image) et par taille d'objet (rappel par objet)
    cat = pd.DataFrame({label[mid]: PROB[mid][2].assign(cat=PROB[mid][2].name.map(category_of)).groupby('cat').iou.mean()
                        for mid in PROB})
    cat['images'] = PROB[BEST['BO2']][2].name.map(category_of).value_counts()
    display(cat.round(3))
    sizes = {}
    for mid, (pt, thr, _) in PROB.items():
        o = object_recall(pt, gt_test, thr, min_cover=0.5, names=names_test)
        o = o[o.area_px >= 4]
        o['bin'] = pd.cut(o.area_px, [0, 100, 400, 1600, 6400, 1e9], labels=['<100', '100-400', '400-1600', '1600-6400', '>6400'])
        sizes[label[mid]] = o.groupby('bin', observed=False).detected.mean()
        sizes['objets'] = o.groupby('bin', observed=False).size()
    sizes = pd.DataFrame(sizes); display(sizes.round(2))
    EVAL['BO2_par_categorie'] = cat.round(4).rename_axis('categorie').reset_index().to_dict('records')
    EVAL['BO2_par_taille'] = sizes.round(4).rename_axis('taille').reset_index().astype({'taille': str}).to_dict('records')
    """),
    md("**Meilleurs et pires cas** du meilleur modèle BO2 (vert = trouvé, rouge = fausse alerte, bleu = manqué) :"),
    code("""
    # Meilleurs et pires cas du meilleur modèle BO2
    pt, thr, pi = PROB[BEST['BO2']]
    order = pi.sort_values('iou').index.tolist()
    sel = order[-3:][::-1] + order[:3]
    show_predictions(DATA['test'][0][sel], pt[sel], gt_test[sel], thr, [names_test[i] for i in sel],
                     n=6, path=os.path.join(FIG, '04_bo2_best_worst.png')); plt.show()
    display(pi.round(3))
    """),
    md("""
## 4. BO3 — Occupation et espace libre du couloir de conduite

Erreur absolue moyenne en points de pourcentage sur les trois sorties, et part des images dont la classe
d'occupation et la classe de passage sont correctes. Le meilleur des trois modèles est choisi sur l'erreur
de validation ; c'est lui qui est déployé.

Référence d'analyse, non déployée : les **mêmes règles du couloir appliquées au masque du meilleur modèle
BO2**, mesurées sur la validation et sur le test, pour savoir si prédire l'occupation directement vaut mieux
que la déduire de la segmentation.
"""),
    code("""
    rows = []
    for s in BO_MODELS['BO3']:
        z = load_pred(s['id'])
        val_mae = float(np.abs(z['p_val'] - z['t_val']).mean())
        rows.append({'id': s['id'], 'modèle': label[s['id']], 'MAE val (pts)': val_mae * 100,
                     **regression_report(z['p_test'], z['t_test']), 'ms / image': TRAIN[s['id']]['ms_per_item']})
    T_TEST = load_pred(BO_MODELS['BO3'][0]['id'])['t_test']
    BO3_SIZE = tuple(load_json(os.path.join(RES, '03_training.json'))['bo3_size'])
    corr = corridor_mask(BO3_SIZE[1], BO3_SIZE[0])
    pt2, thr2, _ = PROB[BEST['BO2']]
    rules = lambda probs: np.stack([corridor_targets(np.asarray(Image.fromarray((p >= thr2).astype(np.uint8))
                                                                .resize(BO3_SIZE, Image.NEAREST)), corr) for p in probs])
    ref = rules(pt2)
    ref_val = rules(load_pred(BEST['BO2'])['val'].astype(np.float32))
    T_VAL = load_pred(BO_MODELS['BO3'][0]['id'])['t_val']
    rows.append({'id': 'bo3_ref_bo2', 'modèle': f'Référence non déployée : règles sur le masque de {label[BEST["BO2"]]}',
                 'MAE val (pts)': float(np.abs(ref_val - T_VAL).mean()) * 100, **regression_report(ref, T_TEST),
                 'ms / image': np.nan})
    bo3 = pd.DataFrame(rows).set_index('id')
    BEST['BO3'] = bo3.drop(index='bo3_ref_bo2')['MAE val (pts)'].idxmin()
    display(bo3.round(3))
    print('Meilleur modèle BO3 (choisi sur la validation) :', label[BEST['BO3']])
    EVAL['BO3'] = bo3.reset_index().to_dict('records')
    np.savez_compressed(os.path.join(PREDS, 'bo3_ref_bo2.npz'), p_test=ref, t_test=T_TEST)
    """),
    md("""
## 5. Résultats : tableau de synthèse

Score de test des trois modèles de chaque BO (F1 pour BO1, IoU pour BO2, erreur sur l'occupation pour
BO3) ; le meilleur, choisi sur la validation, est indiqué dans la dernière colonne.
"""),
    code("""
    THRESH = {'BO1': float(bo1.loc[BEST['BO1'], 'seuil']), 'BO2': float(bo2.loc[BEST['BO2'], 'seuil'])}
    summary = pd.DataFrame([
        {'BO': 'BO1', 'meilleur modèle': label[BEST['BO1']], 'métrique test': 'F1', 'valeur': bo1.loc[BEST['BO1'], 'f1']},
        {'BO': 'BO2', 'meilleur modèle': label[BEST['BO2']], 'métrique test': 'IoU', 'valeur': bo2.loc[BEST['BO2'], 'iou']},
        {'BO': 'BO3', 'meilleur modèle': label[BEST['BO3']], 'métrique test': 'erreur occupation (pts)',
         'valeur': bo3.loc[BEST['BO3'], 'mae_occupation_pts']}])
    display(summary.round(3))
    METRIC = {'BO1': ('f1', bo1), 'BO2': ('iou', bo2), 'BO3': ('mae_occupation_pts', bo3.drop(index='bo3_ref_bo2'))}
    table = pd.DataFrame([{'BO': bo, **{c: f"{s['name']} : {float(df.loc[s['id'], k]):.3f}"
                                         for c, s in zip(['modèle 1 (from scratch)', 'modèle 2 (from scratch)',
                                                          'modèle 3 (pré-entraîné)'], BO_MODELS[bo])},
                           'meilleur (validation)': label[BEST[bo]]}
                          for bo, (k, df) in METRIC.items()]).set_index('BO')
    display(table)
    save_json({'notebook': '04', 'env': ENV, 'best': BEST, 'thresholds': THRESH, 'labels': label, 'results': EVAL},
              os.path.join(RES, '04_evaluation.json'))
    """),
    md("""
## Analyse critique

*Rédigée à partir des sorties de l'exécution, dans la copie exécutée de ce notebook
(`outputs/executed/04_evaluation_executed.ipynb`).*
"""),
]

write("04_evaluation.ipynb", nb04)
