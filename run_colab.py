"""
run_colab.py — exécute toute la chaîne (notebooks 01 à 05) sur Google Colab, sans Drive.

Dans une cellule Colab, après avoir déposé RoadAnomaly_BO_Project.zip dans /content :
    import zipfile; zipfile.ZipFile('/content/RoadAnomaly_BO_Project.zip').extractall('/content')
    exec(open('/content/RoadAnomaly_BO_Project/run_colab.py').read())
Variables : RA_QUICK=1 pour le mode test rapide (2 époques).
"""
import json
import os
import subprocess
import sys
import time

P = "/content/RoadAnomaly_BO_Project"
os.environ["RA_PROJECT_DIR"] = P
os.environ.setdefault("RA_QUICK", "0")
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout, flush=True)
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "papermill", "onnx", "onnxruntime", "onnxscript"], check=True)
import papermill as pm  # noqa: E402

NB = P + "/notebooks"
OUT = P + ("/outputs_quick" if os.environ["RA_QUICK"] == "1" else "/outputs") + "/executed"
os.makedirs(OUT, exist_ok=True)
STATUS = {}
for nb in ["01_exploration_dataset", "02_preprocessing_splits", "03_modelisation", "04_evaluation", "05_deploiement"]:
    t = time.time()
    try:
        pm.execute_notebook(f"{NB}/{nb}.ipynb", f"{OUT}/{nb}_executed.ipynb", cwd=NB, kernel_name="python3",
                            progress_bar=False)
        STATUS[nb] = f"ok {time.time() - t:.0f}s"
    except Exception as e:  # la cellule fautive est visible dans le notebook exécuté
        STATUS[nb] = "ECHEC " + str(e).strip().splitlines()[-1][:300]
        print(nb, STATUS[nb], flush=True)
        break
    print(nb, STATUS[nb], flush=True)
print(json.dumps(STATUS, indent=1, ensure_ascii=False))
