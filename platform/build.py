"""Construit la plateforme métier à partir du vrai report_pack.json (sortie du notebook 05)."""
import base64, io, json, os, sys, zlib
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import load_image, load_mask

ROOT = os.environ.get('RA_DATA', 'data/RoadAnomaly')
pack = json.load(open(sys.argv[1]))
out = sys.argv[2]
H, W = pack['mask_shape']
best = pack['evaluation']['best']['BO2']
names = pack['test_names']


def pack_mask(m):
    return base64.b64encode(zlib.compress(np.packbits(m.astype(bool).ravel()).tobytes(), 9)).decode()


images, refs = [], []
for n in names:
    im = load_image(ROOT, n, (W, H))
    buf = io.BytesIO(); Image.fromarray(im).save(buf, 'JPEG', quality=82)
    images.append('data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode())
    refs.append(pack_mask(load_mask(ROOT, n, (W, H)) > 0))

scenes = pack['scenes']
assert [s['fichier'] for s in scenes] == names
kpi = dict(pack['kpi'])
print('passage', kpi['passage_correct'], 'risque', kpi['risque_correct'])
data = {'mask_shape': [H, W], 'kpi': kpi, 'threshold': pack['evaluation']['thresholds']['BO2'], 'scenes': scenes, 'images': images,
        'pred': pack['masks_test'][best], 'ref': refs,
        'footer': f"Résultats issus de l'exécution réelle du projet (10 photos de contrôle du jeu RoadAnomaly). "
                  f"Analyse en navigateur : {pack['export']['int8']['ms_cpu']:.0f} ms par photo."}
html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'template.html')).read().replace('/*__DATA__*/null', json.dumps(data, ensure_ascii=False))
open(out, 'w').write(html)
print(out, round(len(html) / 1e3), 'ko')
