import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import LSTMAutoencoder


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_station(
    station_dir: Path,
    device: torch.device,
    epochs: int = 5
):

    X_train = np.load(station_dir / "X_train.npy")
    X_val = np.load(station_dir / "X_val.npy")

    meta = json.loads((station_dir / "meta.json").read_text(encoding="utf-8"))

    n_features = X_train.shape[2]
    cohort = meta["cohort"]

    if cohort == "A":
        hidden = 64
        layers = 2
        batch_size = 256
    elif cohort == "B":
        hidden = 32
        layers = 1
        batch_size = 512
    else:
        hidden = 16
        layers = 1
        batch_size = 512

    print(f"\n=== TRAIN {station_dir.name} ===")
    print(f"device={device} | cohort={cohort} | features={n_features} | hidden={hidden} | layers={layers} | batch={batch_size}")

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = LSTMAutoencoder(
        n_features=n_features,
        hidden_size=hidden,
        num_layers=layers
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_epoch = 0

    train_log = []

    for epoch in range(1, epochs + 1):

        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Train {station_dir.name} Epoch {epoch}/{epochs}")

        for (x,) in pbar:
            x = x.to(device)

            optimizer.zero_grad()

            recon = model(x)
            loss = criterion(recon, x)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pbar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                recon = model(x)
                loss = criterion(recon, x)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"[{station_dir.name}] Epoch {epoch}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if device.type == "cuda":
            mem = torch.cuda.memory_allocated(device) / 1024 ** 3
            print(f"[{station_dir.name}] GPU memory used: {mem:.2f} GB")

        train_log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch

            torch.save({
                "model_state_dict": model.state_dict(),
                "n_features": n_features,
                "hidden_size": hidden,
                "num_layers": layers,
                "best_epoch": best_epoch,
                "best_val_loss": best_val
            }, station_dir / "model.pt2")

    (station_dir / "train_log.json").write_text(
        json.dumps(train_log, indent=2, ensure_ascii=False)
    )

    print(f"[OK] saved best model -> {station_dir / 'model.pt2'}")
    print(f"[OK] saved train log  -> {station_dir / 'train_log.json2'}")


def main():

    pr = project_root()

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)

    args = parser.parse_args()

    device = get_device()

    dataset_root = pr / "data" / "processed" / "datasets"

    stations = sorted([p for p in dataset_root.iterdir() if p.is_dir()])

    if args.limit:
        stations = stations[:args.limit]

    for station_dir in stations:
        train_one_station(
            station_dir,
            device,
            epochs=args.epochs
        )


if __name__ == "__main__":
    main()