from common import code, md, write

nb00 = [md("""
# 00 — Notebook principal : exécution de tout le projet sur Google Colab

Ce notebook exécute les notebooks 01 à 07 **dans l'ordre**, chacun dans un noyau Python neuf
(mémoire GPU libérée entre deux notebooks), grâce à **papermill**. Pour chaque notebook, une copie
**exécutée, avec toutes ses sorties réelles** (tableaux, figures, erreurs éventuelles) est enregistrée dans
`outputs/executed/`. Les notebooks d'origine restent vierges.

| Ordre | Notebook | A besoin de | Produit |
|---|---|---|---|
| 1 | 01_exploration_dataset | — | audit, `01_exploration.json` |
| 2 | 02_preprocessing_splits | — | split, tableaux, patches, `02_preprocessing.json` |
| 3 | 03_modelisation | 02 | 9 modèles entraînés (3 par BO), prédictions, `03_training.json` |
| 4 | 04_evaluation | 03 | comparaison, meilleur modèle par BO, `04_evaluation.json` |
| 5 | 05_deploiement | 04 | modèle ONNX, données de la plateforme, `deploy.zip` |

**Conseil** : lancer d'abord le **mode test rapide** (`QUICK = True` : 2 époques par modèle, sorties séparées
dans `outputs_quick/`) pour vérifier que toute la chaîne s'exécute sans erreur, puis l'exécution complète
(`QUICK = False`). Les chiffres du mode rapide ne sont **pas** des résultats et ne doivent jamais être rapportés.

Durée de l'exécution complète : *à mesurer* (elle est affichée à la fin). Si la session Colab est coupée,
relancez ce notebook avec `SKIP_DONE = True` : les notebooks déjà terminés ne sont pas ré-exécutés.
"""),
    md("## 1. GPU, Drive et dossier du projet"),
    code("""
    !nvidia-smi || echo "Pas de GPU : Exécution > Modifier le type d'exécution > GPU T4"
    """),
    code("""
    import os, sys, time, json, zipfile
    USE_DRIVE = True        # False : projet dans /content (Drive plein ou non souhaité ; fichiers perdus en fin de session)
    if USE_DRIVE:
        from google.colab import drive
        drive.mount('/content/drive')
        PROJECT_DIR = '/content/drive/MyDrive/RoadAnomaly_BO_Project'   # à adapter si besoin
    else:
        PROJECT_DIR = '/content/RoadAnomaly_BO_Project'
    zip_path = PROJECT_DIR + '.zip'
    if not os.path.isdir(os.path.join(PROJECT_DIR, 'src')) and os.path.isfile(zip_path):
        zipfile.ZipFile(zip_path).extractall(os.path.dirname(PROJECT_DIR))
    NB_DIR = os.path.join(PROJECT_DIR, 'notebooks')
    assert os.path.isdir(os.path.join(PROJECT_DIR, 'src')) and os.path.isdir(NB_DIR), \\
        f'Projet introuvable dans {PROJECT_DIR} : copiez-y le dossier RoadAnomaly_BO_Project.'
    print(sorted(f for f in os.listdir(NB_DIR) if f.endswith('.ipynb')))
    """),
    code("""
    !pip -q install papermill
    """),
    md("## 2. Paramètres de l'exécution"),
    code("""
    QUICK = True            # True : test rapide de la chaîne (2 époques) ; False : exécution complète
    SKIP_DONE = True        # True : ne pas ré-exécuter un notebook déjà terminé avec succès
    STOP_ON_ERROR = True    # True : arrêter dès qu'un notebook échoue
    KERNEL = 'python3'

    RUN = {                 # mettre False pour ne pas exécuter un notebook
        '01_exploration_dataset.ipynb': True,
        '02_preprocessing_splits.ipynb': True,
        '03_modelisation.ipynb': True,
        '04_evaluation.ipynb': True,
        '05_deploiement.ipynb': True,
    }
    # Dépendance satisfaite si le fichier de résultats du notebook requis existe.
    MARKER = {'01_exploration_dataset.ipynb': '01_exploration.json',
              '02_preprocessing_splits.ipynb': '02_preprocessing.json',
              '03_modelisation.ipynb': '03_training.json',
              '04_evaluation.ipynb': '04_evaluation.json'}
    NEEDS = {'03_modelisation.ipynb': ['02_preprocessing_splits.ipynb'],
             '04_evaluation.ipynb': ['03_modelisation.ipynb'],
             '05_deploiement.ipynb': ['04_evaluation.ipynb']}

    OUT_DIR = os.path.join(PROJECT_DIR, 'outputs_quick' if QUICK else 'outputs')
    RES_DIR, EXEC_DIR = os.path.join(OUT_DIR, 'results'), os.path.join(OUT_DIR, 'executed')
    os.makedirs(EXEC_DIR, exist_ok=True)
    # Variables lues par la cellule de configuration de chaque notebook (noyaux enfants)
    os.environ['RA_PROJECT_DIR'] = PROJECT_DIR
    os.environ['RA_QUICK'] = '1' if QUICK else '0'
    print('Mode :', 'TEST RAPIDE (résultats non valides)' if QUICK else 'COMPLET', '| sorties :', OUT_DIR)
    """),
    md("## 3. Exécution"),
    code("""
    import papermill as pm
    import pandas as pd

    STATUS = {}
    done = lambda nb: nb in MARKER and os.path.isfile(os.path.join(RES_DIR, MARKER[nb]))

    def run(nb):
        out = os.path.join(EXEC_DIR, nb.replace('.ipynb', '_executed.ipynb'))
        if not RUN.get(nb, False):
            STATUS[nb] = {'statut': 'désactivé'}; return True
        if SKIP_DONE and done(nb) and os.path.isfile(out):
            STATUS[nb] = {'statut': 'déjà terminé (ignoré)', 'sortie': out}; return True
        missing = [d for d in NEEDS.get(nb, []) if not done(d)]
        if missing:
            STATUS[nb] = {'statut': f'non lancé : dépendance manquante {missing}'}; return False
        print(f'\\n========== {nb} ==========')
        t0 = time.time()
        try:
            pm.execute_notebook(os.path.join(NB_DIR, nb), out, cwd=NB_DIR, kernel_name=KERNEL,
                                log_output=True, progress_bar=True)
            STATUS[nb] = {'statut': 'succès', 'minutes': round((time.time() - t0) / 60, 1), 'sortie': out}
            return True
        except Exception as e:   # PapermillExecutionError : la cellule fautive est visible dans `out`
            STATUS[nb] = {'statut': 'ÉCHEC', 'minutes': round((time.time() - t0) / 60, 1),
                          'erreur': str(e).splitlines()[-1][:300] if str(e) else repr(e), 'sortie': out}
            return False

    T0 = time.time()
    for nb in RUN:
        ok = run(nb)
        if not ok and STOP_ON_ERROR and STATUS[nb]['statut'] == 'ÉCHEC':
            print(f'Arrêt : {nb} a échoué (voir {STATUS[nb]["sortie"]}).')
            break
    print(f'Durée totale : {(time.time() - T0) / 60:.1f} min')
    """),
    md("## 4. Bilan"),
    code("""
    display(pd.DataFrame(STATUS).T)
    print('Fichiers de résultats :', sorted(os.listdir(RES_DIR)) if os.path.isdir(RES_DIR) else 'aucun')
    exp = os.path.join(OUT_DIR, 'webapp_models')
    print('Modèles exportés pour l\\'application :', sorted(os.listdir(exp)) if os.path.isdir(exp) else 'aucun')
    """),
    code("""
    # Téléchargement de l'archive de déploiement et des notebooks exécutés
    from google.colab import files
    zp = os.path.join(OUT_DIR, 'deploy.zip')
    nb_zip = os.path.join(OUT_DIR, 'notebooks_executes.zip')
    with zipfile.ZipFile(nb_zip, 'w') as zf:
        for f in os.listdir(EXEC_DIR): zf.write(os.path.join(EXEC_DIR, f), f)
    for f in [zp, nb_zip]:
        if os.path.isfile(f): files.download(f)
    """),
    md("""
## 5. Après l'exécution complète (`QUICK = False`)
- Ouvrir les notebooks exécutés dans `outputs/executed/` : ce sont eux qui contiennent les sorties réelles à
  rendre et à citer dans le rapport. Rédiger les sections « à rédiger après exécution ».
- `04_evaluation_executed.ipynb` rassemble la comparaison des modèles ; `outputs/deploy.zip` contient le
  modèle déployé et les données de la plateforme.
- Remplir les gabarits manuels (`outputs/results/*_template.csv`) puis relancer seulement le notebook concerné
  (mettre les autres à `False` dans `RUN`, ou `SKIP_DONE = True`).
- Application : copier `outputs/webapp_models/*` dans `webapp/models/` (voir `webapp/README_webapp.md`).
"""),
]

write("00_main.ipynb", nb00)
