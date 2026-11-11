"""models_catalog.py — catalogue des 9 modèles (3 par BO), lisible sans PyTorch."""

BO_MODELS = {
    "BO1": [
        {"id": "bo1_cnn", "name": "CNN à 4 blocs", "init": "from scratch", "kind": "classifier"},
        {"id": "bo1_ae", "name": "Autoencodeur convolutionnel débruiteur", "init": "from scratch", "kind": "autoencoder"},
        {"id": "bo1_resnet50", "name": "ResNet-50", "init": "pré-entraîné ImageNet", "kind": "classifier"},
    ],
    "BO2": [
        {"id": "bo2_unet", "name": "U-Net", "init": "from scratch", "kind": "segmentation"},
        {"id": "bo2_resunet_scratch", "name": "U-Net à encodeur ResNet-34", "init": "from scratch", "kind": "segmentation"},
        {"id": "bo2_resunet_imagenet", "name": "U-Net à encodeur ResNet-34", "init": "pré-entraîné ImageNet", "kind": "segmentation"},
    ],
    "BO3": [
        {"id": "bo3_cnn", "name": "CNN régresseur", "init": "from scratch", "kind": "regressor"},
        {"id": "bo3_mlp", "name": "Réseau dense (MLP)", "init": "from scratch", "kind": "regressor"},
        {"id": "bo3_resnet18", "name": "ResNet-18", "init": "pré-entraîné ImageNet", "kind": "regressor"},
    ],
}


