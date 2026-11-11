"""
train.py — Boucles d'entraînement / validation, early stopping, sauvegarde et export.

Principes (identiques pour tous les modèles, pour une comparaison équitable) :
    * sélection du meilleur état sur la VALIDATION uniquement (le test n'est vu qu'à la fin) ;
    * early stopping (patience) ; historique complet renvoyé pour tracer les courbes ;
    * précision mixte (AMP) sur GPU ; graines fixées par config.set_seed avant l'appel.
"""
from __future__ import annotations

import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class EarlyStopping:
    """Arrête l'entraînement si la métrique surveillée ne s'améliore plus pendant `patience` époques."""

    def __init__(self, patience: int = 15, mode: str = "max"):
        self.patience, self.mode = patience, mode
        self.best, self.best_epoch, self.best_state = None, 0, None

    def step(self, value: float, epoch: int, model: nn.Module) -> bool:
        better = self.best is None or (value > self.best if self.mode == "max" else value < self.best)
        if better:
            self.best, self.best_epoch = value, epoch
            self.best_state = copy.deepcopy(model.state_dict())
        return epoch - self.best_epoch >= self.patience  # True -> arrêter


def make_optimizer(model: nn.Module, head_params: list | None, lr_backbone: float, lr_head: float,
                   weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    """AdamW avec deux taux : faible pour les couches pré-entraînées, élevé pour la tête."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not head_params:
        return torch.optim.AdamW(params, lr=lr_head, weight_decay=weight_decay)
    head_ids = {id(p) for p in head_params}
    groups = [{"params": [p for p in params if id(p) not in head_ids], "lr": lr_backbone},
              {"params": [p for p in params if id(p) in head_ids], "lr": lr_head}]
    return torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=weight_decay)


def _amp_enabled(device, amp: bool) -> bool:
    return amp and torch.device(device).type == "cuda"


# --------------------------------------------------------------------------------------
# 1. Classification (BO1 : présence / absence d'anomalie sur des patches)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    """Probabilité de la classe « anomalie » (softmax[:, 1]) et labels vrais."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        logits = model(x.to(device))
        probs.append(logits.float().softmax(1)[:, 1].cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(probs), np.concatenate(labels)


def train_classifier(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device,
                     epochs: int = 30, patience: int = 8, lr_backbone: float = 1e-4, lr_head: float = 1e-3,
                     head_params: list | None = None, class_weights: torch.Tensor | None = None,
                     weight_decay: float = 1e-4, amp: bool = True, log_every: int = 1) -> dict:
    """Entraîne un classifieur ; le meilleur état (F1 de validation) est rechargé à la fin."""
    from .evaluate import binary_classification_metrics
    model.to(device)
    opt = make_optimizer(model, head_params, lr_backbone, lr_head, weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    scaler = torch.amp.GradScaler("cuda", enabled=_amp_enabled(device, amp))
    stopper, hist, t0 = EarlyStopping(patience, "max"), {"train_loss": [], "val_f1": [], "val_auroc": []}, time.time()
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), torch.as_tensor(y).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(torch.device(device).type, enabled=_amp_enabled(device, amp)):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()
        p, yv = predict_proba(model, val_loader, device)
        m = binary_classification_metrics(yv, p, 0.5)
        hist["train_loss"].append(float(np.mean(losses)))
        hist["val_f1"].append(m["f1"])
        hist["val_auroc"].append(m["auroc"])
        if ep % log_every == 0:
            print(f"ép {ep:3d}  loss {hist['train_loss'][-1]:.4f}  val F1 {m['f1']:.3f}  "
                  f"AUROC {m['auroc']:.3f}  ({time.time() - t0:.0f}s)")
        if stopper.step(m["f1"], ep, model):
            break
    model.load_state_dict(stopper.best_state)
    hist.update(best_epoch=stopper.best_epoch, best_val_f1=stopper.best, seconds=time.time() - t0)
    return hist


# --------------------------------------------------------------------------------------
# 2. Autoencodeurs (entraînés sur des patches NORMAUX uniquement)
# --------------------------------------------------------------------------------------
def train_autoencoder(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device,
                      epochs: int = 40, patience: int = 8, lr: float = 1e-3, noise_std: float = 0.0,
                      amp: bool = True) -> dict:
    """Perte MSE de reconstruction. noise_std > 0 -> autoencodeur débruiteur (entrée bruitée,
    cible propre). Les loaders doivent fournir des images dans [0, 1] (normalize_imagenet=False)."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    stopper, hist = EarlyStopping(patience, "min"), {"train_loss": [], "val_loss": []}
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for x, _ in train_loader:
            x = x.to(device)
            inp = (x + noise_std * torch.randn_like(x)).clamp(0, 1) if noise_std > 0 else x
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(inp), x)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([F.mse_loss(model(x.to(device)), x.to(device)).item() for x, _ in val_loader]))
        hist["train_loss"].append(float(np.mean(losses)))
        hist["val_loss"].append(vl)
        print(f"ép {ep:3d}  loss {hist['train_loss'][-1]:.5f}  val {vl:.5f}")
        if stopper.step(vl, ep, model):
            break
    model.load_state_dict(stopper.best_state)
    hist.update(best_epoch=stopper.best_epoch, best_val_loss=stopper.best)
    return hist


# --------------------------------------------------------------------------------------
# 3. Segmentation binaire (BO2)
# --------------------------------------------------------------------------------------
def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """BCE pondérée (déséquilibre pixels normaux / anormaux) + Dice (chevauchement global)."""
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, target,
                                             pos_weight=torch.tensor([pos_weight], device=logits.device))
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + 1 - dice.mean()


@torch.no_grad()
def predict_masks(model: nn.Module, images: np.ndarray, device, batch_size: int = 4,
                  tta_flip: bool = False) -> np.ndarray:
    """Probabilités d'anomalie (N, H, W) pour des images uint8 (N, H, W, 3)."""
    from .dataset import normalize
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(np.stack([normalize(im) for im in images[i:i + batch_size]])).to(device)
        p = torch.sigmoid(model(x).float())
        if tta_flip:
            p = 0.5 * (p + torch.sigmoid(model(x.flip(3)).float()).flip(3))
        out.append(p[:, 0].cpu().numpy())
    return np.concatenate(out)


def train_segmentation(model: nn.Module, train_loader: DataLoader, val_images: np.ndarray,
                       val_masks: np.ndarray, device, epochs: int = 60, patience: int = 15,
                       lr_backbone: float = 1e-4, lr_head: float = 1e-3, head_params: list | None = None,
                       pos_weight: float = 10.0, weight_decay: float = 1e-4, amp: bool = True,
                       log_every: int = 5) -> dict:
    """Fine-tuning binaire ; sélection du meilleur état sur le Dice de validation (seuil 0,5)."""
    from .evaluate import segmentation_metrics
    model.to(device)
    opt = make_optimizer(model, head_params, lr_backbone, lr_head, weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=_amp_enabled(device, amp))
    stopper, hist, t0 = EarlyStopping(patience, "max"), {"train_loss": [], "val_dice": [], "val_iou": []}, time.time()
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(torch.device(device).type, enabled=_amp_enabled(device, amp)):
                logits = model(x)
            loss = bce_dice_loss(logits, y, pos_weight)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()
        m = segmentation_metrics(predict_masks(model, val_images, device), val_masks, 0.5)
        hist["train_loss"].append(float(np.mean(losses)))
        hist["val_dice"].append(m["dice"])
        hist["val_iou"].append(m["iou"])
        if ep % log_every == 0:
            print(f"ép {ep:3d}  loss {hist['train_loss'][-1]:.4f}  val Dice {m['dice']:.3f}  "
                  f"IoU {m['iou']:.3f}  ({time.time() - t0:.0f}s)")
        if stopper.step(m["dice"], ep, model):
            break
    model.load_state_dict(stopper.best_state)
    hist.update(best_epoch=stopper.best_epoch, best_val_dice=stopper.best, seconds=time.time() - t0)
    return hist


# --------------------------------------------------------------------------------------
# 4. Sauvegarde, chargement, export pour l'application web
# --------------------------------------------------------------------------------------
def save_checkpoint(path: str, model: nn.Module, meta: dict) -> None:
    """torch.save({'state_dict', 'meta'}) ; meta = architecture, seuil, taille d'entrée, versions…"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path: str, model: nn.Module, device="cpu") -> dict:
    """Recharge les poids dans `model` (même architecture) et renvoie les métadonnées."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return ckpt.get("meta", {})


def export_torchscript(model: nn.Module, example_size_wh: tuple, path: str, meta: dict,
                       sigmoid: bool = True) -> None:
    """Exporte le modèle tracé (TorchScript) + un fichier JSON de métadonnées.
    L'application web n'a alors besoin ni du dépôt VainF, ni du code d'architecture.
    Entrée du modèle exporté : image normalisée ImageNet (1, 3, H, W) à la taille fixe
    example_size_wh. Sortie : probabilité (sigmoid=True) ou logits (1, C, H, W)."""

    class _Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            y = self.m(x)
            return torch.sigmoid(y) if sigmoid else y

    w, h = example_size_wh
    wrapped = _Wrap(model).eval().cpu()
    traced = torch.jit.trace(wrapped, torch.zeros(1, 3, h, w))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    traced.save(path)
    meta = dict(meta, input_size_wh=[w, h], output="probability" if sigmoid else "logits",
                normalization="imagenet")
    with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=float)


# --------------------------------------------------------------------------------------
# 5. Régression (BO3 : occupation et espace libre du couloir)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def predict_regression(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    P, T = [], []
    for x, y in loader:
        P.append(model(x.to(device)).float().cpu().numpy())
        T.append(y.numpy())
    return np.concatenate(P), np.concatenate(T)


def train_regressor(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device,
                    epochs: int = 80, patience: int = 15, lr_backbone: float = 1e-4, lr_head: float = 1e-3,
                    head_params: list | None = None, weight_decay: float = 1e-4, log_every: int = 5) -> dict:
    """Perte Smooth L1 sur les 3 sorties ; meilleur état = plus petite erreur absolue moyenne de validation."""
    model.to(device)
    opt = make_optimizer(model, head_params, lr_backbone, lr_head, weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    stopper, hist, t0 = EarlyStopping(patience, "min"), {"train_loss": [], "val_mae": []}, time.time()
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss(model(x).float(), y, beta=0.05)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        p, t = predict_regression(model, val_loader, device)
        mae = float(np.abs(p - t).mean())
        hist["train_loss"].append(float(np.mean(losses)))
        hist["val_mae"].append(mae)
        if ep % log_every == 0:
            print(f"ép {ep:3d}  loss {hist['train_loss'][-1]:.4f}  val MAE {mae:.4f}  ({time.time() - t0:.0f}s)")
        if stopper.step(mae, ep, model):
            break
    model.load_state_dict(stopper.best_state)
    hist.update(best_epoch=stopper.best_epoch, best_val_mae=stopper.best, seconds=time.time() - t0)
    return hist
