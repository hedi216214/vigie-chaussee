"""Générateur des notebooks Colab (format nbformat 4, sans sorties).
Usage : python tools/nbgen/build_all.py  -> écrit notebooks/*.ipynb"""
import copy
import json
import os
import textwrap

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "notebooks")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": textwrap.dedent(text).strip("\n")}


def write(name, cells):
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}},
          "cells": copy.deepcopy(cells)}
    for c in nb["cells"]:  # nbformat attend une liste de lignes
        src = c["source"]
        c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)


INTEGRITY = md("""
> Tous les chiffres, tableaux et figures de ce notebook sont produits par son exécution sur le jeu
> RoadAnomaly ; aucun résultat n'est écrit à la main.
""")


def setup_cells(pip=""):
    cells = [
        md("""
## 0. Préparer l'environnement

On vérifie le GPU, on localise le dossier du projet (Colab ou local), on charge la configuration commune
(graine 42, résolution de travail 512 × 288) et on télécharge RoadAnomaly s'il est absent.
"""),
        code("""
        # Vérification du GPU (Colab : Exécution > Modifier le type d'exécution > GPU T4)
        !nvidia-smi || echo "Pas de GPU visible : les entraînements seront très lents."
        """),
    ]
    if pip:
        cells.append(code(f"""
        # Installation des dépendances absentes de Colab
        !pip -q install {pip}
        """))
    cells.append(code("""
        import os, sys
        IN_COLAB = 'google.colab' in sys.modules
        USE_DRIVE = True   # False : travail dans /content (fichiers perdus à la fin de la session)
        # Mode test rapide (activé par 00_main.ipynb) : 2 époques par modèle, sorties dans outputs_quick/.
        # Sert uniquement à vérifier que la chaîne s'exécute ; ses chiffres ne doivent jamais être rapportés.
        QUICK = os.environ.get('RA_QUICK') == '1'
        EP = lambda n: 2 if QUICK else n

        if os.environ.get('RA_PROJECT_DIR'):            # exécution pilotée par 00_main.ipynb (Drive déjà monté)
            PROJECT_DIR = os.environ['RA_PROJECT_DIR']
        elif IN_COLAB and USE_DRIVE:
            from google.colab import drive
            drive.mount('/content/drive')
            PROJECT_DIR = '/content/drive/MyDrive/RoadAnomaly_BO_Project'
        elif IN_COLAB:
            PROJECT_DIR = '/content/RoadAnomaly_BO_Project'
        else:
            PROJECT_DIR = os.path.abspath('..')        # exécution locale depuis notebooks/
        DATA_DIR = os.path.join(PROJECT_DIR, 'data')   # contient (ou contiendra) RoadAnomaly/

        # Si seul l'archive du projet a été déposée, on la décompresse
        zip_path = PROJECT_DIR + '.zip'
        if not os.path.isdir(os.path.join(PROJECT_DIR, 'src')) and os.path.isfile(zip_path):
            import zipfile
            zipfile.ZipFile(zip_path).extractall(os.path.dirname(PROJECT_DIR))
        assert os.path.isdir(os.path.join(PROJECT_DIR, 'src')), \\
            f"src/ introuvable dans {PROJECT_DIR} : copiez le dossier du projet à cet endroit."
        if PROJECT_DIR not in sys.path:
            sys.path.insert(0, PROJECT_DIR)

        from src.config import ProjectPaths, TrainConfig, set_seed, environment_report, save_json, load_json
        paths = ProjectPaths(PROJECT_DIR, DATA_DIR, outputs_name='outputs_quick' if QUICK else 'outputs')
        cfg = TrainConfig()
        set_seed(cfg.seed)
        W, H = cfg.work_size
        FIG, RES, PREDS, EXPORT = paths.sub('figures'), paths.sub('results'), paths.sub('preds'), paths.sub('webapp_models')
        ENV = environment_report()
        print(ENV)
        if QUICK:
            print('*** MODE TEST RAPIDE : résultats NON valides, écrits dans', paths.outputs, '***')
        """))
    cells.append(code("""
        from src.dataset import ensure_dataset, list_frames
        ROOT = ensure_dataset(DATA_DIR)        # télécharge et décompresse le dataset si absent
        print('Dataset :', ROOT)
        """))
    return cells


def load_split_cells():
    return [code("""
        import numpy as np
        splits = load_json(os.path.join(paths.sub('splits'), 'splits.json'))    # produit par le notebook 02
        z = np.load(os.path.join(paths.sub('processed'), f'arrays_{W}x{H}.npz'))
        DATA = {s: (z[f'{s}_images'], z[f'{s}_masks'], list(z[f'{s}_names'])) for s in ['train', 'val', 'test']}
        PREP = load_json(os.path.join(RES, '02_preprocessing.json'))
        for s, (im, m, n) in DATA.items():
            print(f'{s:5s}', im.shape, m.shape)
        """)]
