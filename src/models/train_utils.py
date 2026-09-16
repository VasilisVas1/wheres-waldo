"""Training loop for one CV fold of the patch classifier."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.patch_dataset import PatchDataset


@dataclass
class FoldHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)


def run_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer=None, device: str = "cpu"):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += ((torch.sigmoid(logits) > 0.5).float() == labels).sum().item()
        n += len(labels)

    return total_loss / n, correct / n


def train_fold(
    model: nn.Module,
    patch_manifest,
    fold_assignment,
    val_fold: int,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> FoldHistory:
    train_ds = PatchDataset(patch_manifest, fold_assignment, val_fold, split="train")
    val_ds = PatchDataset(patch_manifest, fold_assignment, val_fold, split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Positive patches are far rarer than negative ones even after undersampling
    # (~126 positive vs ~760 negative base patches) — weight the loss so the
    # optimizer doesn't just learn to predict "negative" for everything.
    n_pos = (train_ds.rows["label"] == 1).sum()
    n_neg = (train_ds.rows["label"] == 0).sum()
    pos_weight = torch.tensor(n_neg / max(1, n_pos))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = FoldHistory()
    for _ in range(epochs):
        train_loss, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_accuracy.append(val_acc)

    return history
