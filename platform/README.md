# Plateforme métier « Vigie Chaussée »

Page web en français, sans vocabulaire technique, construite à partir des résultats réels :

    RA_DATA=data/RoadAnomaly python platform/build.py outputs/report_pack.json platform/site/index.html

`build.py` lit `report_pack.json` (notebook 05) et les 10 images de test. Les scènes affichent les sorties des
trois modèles déployés (ResNet-50, U-Net ResNet-34 ImageNet, ResNet-18). L'analyse d'une photo dans le navigateur
utilise `model_int8.onnx` de `deploy.zip` (sorties `prob` = BO2, `corridor` = BO3), découpé en
`site/model/part*.wasm` + `model.json`, et onnxruntime-web 1.14.0 (`site/ort/*.wasm`).
Version publiée : https://claude.ai/artifact/SigMnMwvN3eLCY56hoNRrb
