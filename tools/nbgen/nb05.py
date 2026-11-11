from common import INTEGRITY, code, load_split_cells, md, setup_cells, write

nb05 = [md("""
# 05 — Déploiement : exporter la chaîne retenue et préparer la plateforme métier

La chaîne déployée réunit le meilleur modèle de chaque BO, choisi au notebook 04 sur la validation :
BO1 signale la présence d'un obstacle, BO2 dessine la zone, BO3 donne la part de la voie occupée et
l'espace libre, d'où le passage recommandé et le niveau de risque.
"""), md("""
### Étapes de ce notebook

1. Importer les bibliothèques et charger les choix du notebook 04
2. Exporter les modèles BO2 et BO3 en un seul fichier ONNX, puis le compresser (entiers 8 bits)
3. Calculer les sorties des meilleurs modèles sur les images de test
4. Traduire ces sorties en langage métier (passage, niveau de risque) et calculer les indicateurs
5. Prédiction : analyser une photo de bout en bout
6. Préparer le paquet de résultats et l'archive à télécharger
"""), INTEGRITY] + setup_cells(pip="onnx onnxruntime onnxscript") + load_split_cells() + [
    md("## 1. Importer les bibliothèques et charger les choix du notebook 04"),
    code("""
    import time, shutil, zipfile, torch, torch.nn as nn, torch.nn.functional as F, pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image
    from src.config import get_device, IMAGENET_MEAN, IMAGENET_STD
    from src.dataset import normalize, category_of, family_of, extract_patches
    from src.bo_models import build_bo_model
    from src.models_catalog import BO_MODELS
    from src.train import load_checkpoint, predict_masks
    from src.corridor import classes_from_targets, corridor_mask
    from src.evaluate import object_recall, segmentation_metrics

    device = get_device()
    EV = load_json(os.path.join(RES, '04_evaluation.json'))
    BEST, THR = EV['best'], EV['thresholds']
    kind = {s['id']: s['kind'] for lst in BO_MODELS.values() for s in lst}
    for bo in ['BO1', 'BO2', 'BO3']:
        print(bo, '->', EV['labels'][BEST[bo]])
    DEP = paths.sub('deploy'); PLAT = os.path.join(DEP, 'platform'); os.makedirs(PLAT, exist_ok=True)
    """),
    md("""
## 2. Exporter les modèles BO2 et BO3 en ONNX

ONNX est un format standard qui permet d'exécuter un réseau hors de PyTorch, par exemple dans un
navigateur web. On exporte **un seul graphe** qui prend une photo RGB 512 × 288 (valeurs entre 0 et 1) et
renvoie deux sorties :
- `prob` : la carte « obstacle » du meilleur modèle BO2 (probabilité par pixel) ;
- `corridor` : les trois valeurs du meilleur modèle BO3 (occupation, libre à gauche, libre à droite).

La normalisation ImageNet, la sigmoïde et la réduction de l'image à 256 × 144 pour BO3 sont incluses dans
le graphe. Le fichier est ensuite compressé en entiers 8 bits (4 fois plus petit), et l'on vérifie que la
version compressée donne les mêmes décisions que PyTorch.
"""),
    code("""
    def load_best(bo):
        m = build_bo_model(BEST[bo]); load_checkpoint(os.path.join(paths.sub('checkpoints'), f"{BEST[bo]}.pth"), m)
        return m.eval()

    seg, reg = load_best('BO2').cpu(), load_best('BO3').cpu()

    class Deploy(nn.Module):
        # entrée : RGB 0–1 (1, 3, 288, 512) ; sorties : prob (1, 1, 288, 512) et corridor (1, 3)
        def __init__(self, seg, reg):
            super().__init__(); self.seg, self.reg = seg, reg
            self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
            k = torch.tensor([1., 3., 3., 1.]) / 8                          # = réduction bilinéaire × 1/2 de PIL
            self.register_buffer('down', (k[:, None] * k[None, :]).expand(3, 1, 4, 4).clone())
        def forward(self, x):
            prob = torch.sigmoid(self.seg((x - self.mean) / self.std))
            small = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), self.down, stride=2, groups=3)  # 256 × 144
            return prob, self.reg((small - self.mean) / self.std)

    import onnx, onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    fp32, int8 = os.path.join(DEP, 'model_fp32.onnx'), os.path.join(DEP, 'model_int8.onnx')
    dummy = torch.from_numpy(DATA['test'][0][:1].transpose(0, 3, 1, 2).astype(np.float32) / 255)
    torch.onnx.export(Deploy(seg, reg).eval(), dummy, fp32, input_names=['image'], output_names=['prob', 'corridor'],
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(fp32)
    quantize_dynamic(fp32, int8, weight_type=QuantType.QUInt8)

    def run_onnx(path):
        s = ort.InferenceSession(path, providers=['CPUExecutionProvider']); P, C, t = [], [], time.time()
        for im in DATA['test'][0]:
            p, c = s.run(None, {'image': im.transpose(2, 0, 1)[None].astype(np.float32) / 255})
            P.append(p[0, 0]); C.append(c[0])
        return np.stack(P), np.stack(C), (time.time() - t) / len(P) * 1000

    ref = np.load(os.path.join(PREDS, f"{BEST['BO2']}.npz"))['test'].astype(np.float32)
    ref3 = np.load(os.path.join(PREDS, f"{BEST['BO3']}.npz"))['p_test']
    EXPORT_INFO = {}
    for name, path in [('fp32', fp32), ('int8', int8)]:
        p, c, ms = run_onnx(path)
        m = segmentation_metrics(p, DATA['test'][1], THR['BO2'])
        EXPORT_INFO[name] = {'iou': m['iou'], 'precision': m['precision'], 'recall': m['recall'], 'ms_cpu': ms,
                             'size_MB': os.path.getsize(path) / 1e6, 'max_abs_diff': float(np.abs(p - ref).max()),
                             'decision_agreement': float(((p >= THR['BO2']) == (ref >= THR['BO2'])).mean()),
                             'bo3_ecart_pts': float(np.abs(c - ref3).mean() * 100),
                             'bo3_passage_identique': float(np.mean([classes_from_targets(a)['passabilite'] ==
                                                                     classes_from_targets(b)['passabilite'] for a, b in zip(c, ref3)]))}
    display(pd.DataFrame(EXPORT_INFO).T.round(4))
    """),
    md("""
## 3. Calculer les sorties des meilleurs modèles sur les images de test

- **BO1** : la photo est découpée en zones de 96 × 96 (pas de 48) ; la scène est signalée si au moins une
  zone dépasse le seuil du meilleur modèle BO1.
- **BO2** : les cartes « obstacle » du meilleur modèle (les mêmes prédictions qu'au notebook 04).
- **BO3** : les trois valeurs du meilleur modèle (occupation, libre à gauche, libre à droite).
"""),
    code("""
    ims, gts, names = DATA['test']
    # BO1 : une scène est signalée si au moins une zone de 96 × 96 dépasse le seuil du meilleur modèle BO1
    bo1 = load_best('BO1').to(device)

    @torch.no_grad()
    def zone_scores(img):
        H_, W_ = img.shape[:2]; tiles = []
        for y0 in range(0, H_ - 96 + 1, 48):
            for x0 in range(0, W_ - 96 + 1, 48):
                tiles.append(img[y0:y0 + 96, x0:x0 + 96])
        tiles = np.stack(tiles)
        if kind[BEST['BO1']] == 'autoencoder':
            x = torch.from_numpy(tiles.astype(np.float32).transpose(0, 3, 1, 2) / 255).to(device)
            return ((bo1(x) - x) ** 2).flatten(1).mean(1).cpu().numpy()
        size = 224 if BEST['BO1'] == 'bo1_resnet50' else 96
        x = torch.from_numpy(np.stack([normalize(np.asarray(Image.fromarray(t).resize((size, size)))) for t in tiles])).to(device)
        return bo1(x).float().softmax(1)[:, 1].cpu().numpy()

    BO1_OUT = [zone_scores(im) for im in ims]
    # BO2 : masques du meilleur modèle (mêmes prédictions que l'évaluation) ; BO3 : sorties du meilleur modèle
    P2 = np.load(os.path.join(PREDS, f"{BEST['BO2']}.npz"))['test'].astype(np.float32)
    z3 = np.load(os.path.join(PREDS, f"{BEST['BO3']}.npz")); P3, T3 = z3['p_test'], z3['t_test']
    """),
    md("""
## 4. Traduire en langage métier

Niveau de risque (règle de présentation, hypothèse documentée) : *voie libre* → faible ; *passage des
deux côtés* → modéré ; *passage à gauche / à droite* → élevé ; *passage difficile* ou *voie bloquée* → critique.
"""),
    code("""
    RISK = {'voie libre': 'faible', 'passage des deux côtés': 'modéré', 'passage à gauche': 'élevé',
            'passage à droite': 'élevé', 'passage difficile': 'critique', 'voie bloquée': 'critique'}
    TYPE = {'animals': 'Animal', 'cones': 'Cônes de signalisation', 'obstacles': 'Obstacle sur la chaussée',
            'lost': 'Objet perdu', 'vehicle': 'Véhicule inhabituel'}
    scenes = []
    for i, n in enumerate(names):
        Image.fromarray(ims[i]).save(os.path.join(PLAT, f'img_{i}.jpg'), quality=90)
        Image.fromarray((P2[i] * 255).round().astype(np.uint8)).save(os.path.join(PLAT, f'zone_{i}.png'))
        Image.fromarray((gts[i] > 0).astype(np.uint8) * 255).save(os.path.join(PLAT, f'verite_{i}.png'))
        cp, ct = classes_from_targets(P3[i]), classes_from_targets(T3[i])
        m = segmentation_metrics(P2[i][None], gts[i][None], THR['BO2'])
        scenes.append({'i': i, 'fichier': n, 'type': TYPE[category_of(n)],
                       'obstacle_signale': bool(BO1_OUT[i].max() >= THR['BO1']),
                       'zones_signalees_pct': float((BO1_OUT[i] >= THR['BO1']).mean() * 100),
                       'occupation_pct': float(P3[i][0] * 100), 'libre_gauche_pct': float(P3[i][1] * 100),
                       'libre_droite_pct': float(P3[i][2] * 100), 'classe_occupation': cp['occupation'],
                       'passage': cp['passabilite'], 'risque': RISK[cp['passabilite']],
                       'reel': {'occupation_pct': float(T3[i][0] * 100), 'passage': ct['passabilite'],
                                'risque': RISK[ct['passabilite']]},
                       'zone_recouvrement': m['iou'], 'zone_obstacle_trouve_pct': m['recall'] * 100,
                       'zone_fausse_alerte_pct': (1 - m['precision']) * 100 if m['tp'] + m['fp'] else 0.0})
    pd.DataFrame(scenes).drop(columns='reel')
    """),
    code("""
    obj = object_recall(P2, gts, THR['BO2'], min_cover=0.5, names=names); obj = obj[obj.area_px >= 4]
    m_all = segmentation_metrics(P2, gts, THR['BO2'])
    KPI = {'scenes': len(names),
           'scenes_signalees': int(sum(s['obstacle_signale'] for s in scenes)),
           'obstacles_trouves': int(obj.detected.sum()), 'obstacles_total': int(len(obj)),
           'surface_obstacle_trouvee_pct': m_all['recall'] * 100,
           'surface_signalee_a_tort_pct': (1 - m_all['precision']) * 100,
           'passage_correct': int(sum(s['passage'] == s['reel']['passage'] for s in scenes)),
           'risque_correct': int(sum(s['risque'] == s['reel']['risque'] for s in scenes)),
           'temps_analyse_ms_gpu': load_json(os.path.join(RES, '03_training.json'))['models'][BEST['BO2']]['ms_per_item'],
           'temps_analyse_ms_navigateur_cpu': EXPORT_INFO['int8']['ms_cpu']}
    KPI
    """),
    md("""
## 5. Prédiction : analyser une photo de bout en bout

La fonction `analyser_photo` enchaîne les trois modèles retenus sur une photo quelconque et rend la réponse
en langage métier, comme la plateforme. On l'applique à trois images de test.
"""),
    code("""
    seg_d, reg_d = seg.to(device).eval(), reg.to(device).eval()

    @torch.no_grad()
    def analyser_photo(img_uint8):
        img = np.asarray(Image.fromarray(img_uint8).resize((W, H), Image.BILINEAR))        # 512 × 288
        alerte = bool(zone_scores(img).max() >= THR['BO1'])                                 # BO1
        x = torch.from_numpy(normalize(img))[None].to(device)
        zone = torch.sigmoid(seg_d(x))[0, 0].float().cpu().numpy() >= THR['BO2']          # BO2
        small = np.asarray(Image.fromarray(img).resize(BO3_SIZE, Image.BILINEAR))
        occ, gauche, droite = reg_d(torch.from_numpy(normalize(small))[None].to(device))[0].float().cpu().numpy()  # BO3
        c = classes_from_targets([occ, gauche, droite])
        return {'obstacle signalé': 'oui' if alerte else 'non', 'zone repérée (% de la photo)': round(float(zone.mean()) * 100, 1),
                'voie occupée (%)': round(float(occ) * 100, 1), 'libre à gauche (%)': round(float(gauche) * 100, 1),
                'libre à droite (%)': round(float(droite) * 100, 1), 'passage recommandé': c['passabilite'],
                'niveau de risque': RISK[c['passabilite']]}, zone

    BO3_SIZE = tuple(load_json(os.path.join(RES, '03_training.json'))['bo3_size'])
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
    for ax, i in zip(axes, [0, 2, 6]):
        rep, zone = analyser_photo(ims[i])
        over = ims[i].copy(); over[zone] = (0.45 * over[zone] + 0.55 * np.array([240, 168, 24])).astype(np.uint8)
        ax.imshow(over); ax.axis('off')
        ax.set_title(f"{rep['niveau de risque']} · {rep['passage recommandé']} · voie occupée {rep['voie occupée (%)']} %", fontsize=9)
        print(names[i], rep)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, '05_demo_prediction.png'), dpi=120); plt.show()
    """),
    md("## 6. Préparer le paquet de résultats (rapport et plateforme)"),
    code("""
    # Toutes les métriques + courbes ROC/PR échantillonnées + masques binaires compressés (bits + zlib + base64)
    import base64, zlib
    from sklearn.metrics import roc_curve, precision_recall_curve
    def pack_mask(m): return base64.b64encode(zlib.compress(np.packbits(m.astype(bool).ravel()).tobytes(), 9)).decode()
    def curve(y, s, n=120, max_pts=2_000_000):
        y, s = np.asarray(y).ravel(), np.asarray(s, np.float32).ravel()
        if y.size > max_pts:
            idx = np.random.default_rng(0).choice(y.size, max_pts, replace=False); y, s = y[idx], s[idx]
        f, t, _ = roc_curve(y, s); p, r, _ = precision_recall_curve(y, s)
        i1, i2 = np.linspace(0, len(f) - 1, n).astype(int), np.linspace(0, len(p) - 1, n).astype(int)
        return {'fpr': f[i1].round(4).tolist(), 'tpr': t[i1].round(4).tolist(), 'precision': p[i2].round(4).tolist(), 'recall': r[i2].round(4).tolist()}
    curves = {}
    for s_ in BO_MODELS['BO1']:
        z = np.load(os.path.join(PREDS, f"{s_['id']}.npz")); curves[s_['id']] = curve(z['y_test'], z['s_test'])
    masks = {}
    for s_ in BO_MODELS['BO2']:
        pt = np.load(os.path.join(PREDS, f"{s_['id']}.npz"))['test'].astype(np.float32)
        curves[s_['id']] = curve(DATA['test'][1] > 0, pt)
        thr_m = next(r['seuil'] for r in EV['results']['BO2'] if r['id'] == s_['id'])
        masks[s_['id']] = [pack_mask(p >= thr_m) for p in pt]
    bo3 = {s_['id']: {k: v.round(4).tolist() for k, v in np.load(os.path.join(PREDS, f"{s_['id']}.npz")).items()}
           for s_ in BO_MODELS['BO3'] + [{'id': 'bo3_ref_bo2'}]}
    PACK = {'evaluation': EV, 'training': load_json(os.path.join(RES, '03_training.json')),
            'preprocessing': load_json(os.path.join(RES, '02_preprocessing.json')),
            'export': EXPORT_INFO, 'kpi': KPI, 'scenes': scenes, 'curves': curves, 'masks_test': masks,
            'bo3_predictions': bo3, 'test_names': list(names), 'mask_shape': [H, W]}
    save_json(PACK, os.path.join(paths.outputs, 'report_pack.json'))
    print(round(os.path.getsize(os.path.join(paths.outputs, 'report_pack.json')) / 1e3), 'ko')
    """),
    md("### Archive à télécharger"),
    code("""
    save_json({'best': BEST, 'thresholds': THR, 'labels': EV['labels'], 'export': EXPORT_INFO, 'kpi': KPI,
               'scenes': scenes, 'model': {'input': [1, 3, 288, 512], 'input_range': '0-1 RGB',
                                           'outputs': {'prob': 'BO2', 'corridor': 'BO3'}, 'threshold': THR['BO2']}},
              os.path.join(PLAT, 'data.json'))
    zpath = os.path.join(paths.outputs, 'deploy.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_STORED) as zf:
        zf.write(int8, 'model_int8.onnx')
        for f in os.listdir(PLAT):
            zf.write(os.path.join(PLAT, f), f'platform/{f}')
        for f in os.listdir(RES):
            if f.endswith(('.json', '.csv')): zf.write(os.path.join(RES, f), f'results/{f}')
        for f in os.listdir(FIG):
            if f.startswith(('03_', '04_')): zf.write(os.path.join(FIG, f), f'figures/{f}')
        for f in os.listdir(PREDS):
            if f.startswith(('bo1_', 'bo3_')): zf.write(os.path.join(PREDS, f), f'preds/{f}')
    print(zpath, round(os.path.getsize(zpath) / 1e6, 1), 'Mo')
    if IN_COLAB and not os.environ.get('RA_PROJECT_DIR'):   # exécution directe ; sinon 00_main télécharge
        from google.colab import files
        files.download(zpath)
    """),
]

write("05_deploiement.ipynb", nb05)
