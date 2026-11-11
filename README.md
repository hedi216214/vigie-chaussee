# Vigie Chaussée — Road obstacle detection and passability (Deep Learning)

From a single photo taken facing the road, Vigie Chaussée answers three questions for the driver: **is there an unexpected obstacle, where exactly is it, and is there still room to pass?** The best model of each objective is chained and deployed in a **serverless web platform** that runs entirely in the browser.

Academic project — ESPRIT, 5th year Business Intelligence & Big Data (class 5 ERP-BI 2), Deep Learning module, November 2026. Method: **CRISP-DM**. Dataset: **RoadAnomaly** (Lis et al., ICCV 2019).

## Results

For each business objective, two models trained from scratch were compared with one model pre-trained on ImageNet, on the same data and seed. The winner was chosen on the validation set, then measured once on the test set.

| Objective | Question | From scratch | Pre-trained (deployed) | Test result |
| --- | --- | --- | --- | --- |
| BO1 · Detect | Is there an obstacle? | 4-block CNN (F1 0.78) · denoising autoencoder (F1 0.43) | **ResNet-50** | F1 0.82 · AUROC 0.94 |
| BO2 · Localize | Where is it, pixel by pixel? | U-Net (IoU 0.33) · U-Net ResNet-34, random init (IoU 0.41) | **U-Net ResNet-34 (ImageNet)** | IoU 0.70 · Dice 0.82 |
| BO3 · Pass | How much of the lane is occupied? | CNN regressor (8.6 pts) · MLP (13.6 pts) | **ResNet-18** | 4.8 pts occupancy error |

**Key finding:** with only 60 annotated images, transfer learning matters more than architecture. The same U-Net ResNet-34 goes from IoU 0.41 to 0.70 when its encoder starts from ImageNet weights.

Business evaluation on 10 control scenes: an obstacle is flagged in 10/10 scenes, 25/35 obstacles are found, the risk level matches the ground truth 5/10 times, and the recommended side is right 2/10 times (main limitation).

## Deployment

```
Photo ──► BO1 ResNet-50        alert if one 96×96 tile scores ≥ 0.11
      ──► BO2 U-Net ResNet-34  obstacle mask, pixels ≥ 0.60
      ──► BO3 ResNet-18        lane occupancy, free space left / right
      ──► Risk level           low · moderate · high · critical
```

- BO2 and BO3 are exported together as **one ONNX graph** (input: RGB photo 512×288; outputs: `prob`, `corridor`), with ImageNet normalization, sigmoid and downscaling built into the graph.
- **Dynamic int8 quantization** of the weights: 143 MB → 36 MB, test IoU 0.701 → 0.703, identical decisions on 99.8% of pixels, same passage on 10/10 test images.
- The platform runs the model with **onnxruntime-web (WebAssembly)**: 2–3 s per photo on CPU, no server, the photo never leaves the user's device. Training ran on a Colab T4 GPU (27 ms per image for BO2).

## Repository

| Folder | Content |
| --- | --- |
| `notebooks/` | 01 exploration · 02 preprocessing and splits · 03 modeling (9 models) · 04 evaluation · 05 deployment |
| `outputs/executed/` | Executed notebooks with real outputs |
| `outputs/results/`, `outputs/figures/` | Metrics (JSON/CSV) and figures |
| `src/` | Dataset, models, training, evaluation, lane corridor and passability logic |
| `platform/` | Business platform builder (`build.py`, `template.html`) |
| `vigie chaussee  plateforme/` | Ready-to-open platform with the int8 ONNX model |
| `demo/` | Demo video of the platform |
| `tests/` | Unit tests for BO3 and metrics |

## Run it

**Notebooks (Google Colab, T4 GPU):** upload the project, then run `run_colab.py` or open `notebooks/00_main.ipynb`. The RoadAnomaly dataset is downloaded automatically.

**Platform (local):**

```bash
cd "vigie chaussee  plateforme"
python -m http.server 8000
# open http://localhost:8000
```

An internet connection is needed the first time (onnxruntime-web is loaded from a CDN).

## Limitations and next steps

- Only 10 test images, no obstacle-free test scene, fixed driving corridor (centered camera, straight road).
- Small distant objects and unusual vehicles are often missed.
- Next steps: compute passability from the BO2 mask (already 7/10 correct on test), evaluate on Fishyscapes, SegmentMeIfYouCan and Lost and Found, segment the road to follow curves.

## Team — Deep Detectors

Mohamed Hedi Chebil · Ranin Baccouri · Hedir Zraga · Safwen Bechaouech · Yahya Ghezail · Aziz Cherni

Supervised by Mustapha Trabelsi, ESPRIT.

*Academic demonstrator: not intended for real driving decisions.*
