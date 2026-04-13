# podzial musi byc po folder_id a nie po patchach bo z jednego olderu mamy wiecej patchy

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from src.dataset.patch_dataset import MethanePatchDataset
from src.models.simple_cnn_patch_classifier import SimplePatchCNN


def make_splits(meta_path, test_size=0.2, random_state=42):
    meta = pd.read_csv(meta_path)

    unique_folders = meta["folder_id"].unique()

    train_folders, val_folders = train_test_split(
        unique_folders,
        test_size=test_size,
        random_state=random_state
    )

    train_idx = meta[meta["folder_id"].isin(train_folders)].index.tolist()
    val_idx = meta[meta["folder_id"].isin(val_folders)].index.tolist()

    return train_idx, val_idx


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    losses = []
    all_preds = []
    all_targets = []

    with torch.set_grad_enabled(is_train):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            losses.append(loss.item())

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    avg_loss = np.mean(losses)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")

    return avg_loss, acc, f1


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    x_path = "outputs/dataset/X_data.npy"
    y_path = "outputs/dataset/y_data.npy"
    meta_path = "outputs/dataset/meta_data.csv"

    dataset = MethanePatchDataset(
        x_path=x_path,
        y_path=y_path,
        meta_path=meta_path
    )

    train_idx, val_idx = make_splits(meta_path, test_size=0.2, random_state=42)

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    model = SimplePatchCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20

    best_val_f1 = -1.0
    out_dir = Path("outputs/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = out_dir / "best_patch_cnn.pt"

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_f1 = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer
        )

        val_loss, val_acc, val_f1 = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print("Saved best model.")

if __name__ == "__main__":
    main()