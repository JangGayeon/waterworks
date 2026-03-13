import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import LSTMAutoencoder


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_params_by_cohort(cohort: str) -> Tuple[int, int]:
    """
    cohort별 모델 크기
    A: feature 많음 → 조금 더 큰 모델
    B: 중간
    C: 작게
    """
    cohort = str(cohort).upper()
    if cohort == "A":
        return 64, 2
    if cohort == "B":
        return 32, 1
    return 16, 1


def get_batch_size_by_cohort(cohort: str, default_batch_size: int) -> int:
    cohort = str(cohort).upper()
    if default_batch_size > 0:
        return default_batch_size

    if cohort == "A":
        return 256
    if cohort == "B":
        return 512
    return 512


def build_dataloaders(
    station_dir: Path,
    batch_size: int,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    X_train = np.load(station_dir / "X_train.npy")
    X_val = np.load(station_dir / "X_val.npy")
    meta = load_json(station_dir / "meta.json")

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    val_tensor = torch.tensor(X_val, dtype=torch.float32)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader, meta


def evaluate(model, loader, criterion, device, use_amp: bool = True) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    amp_enabled = use_amp and device.type == "cuda"

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for batch in pbar:
            x = batch[0].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=amp_enabled):
                recon = model(x)
                loss = criterion(recon, x)

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")

    if num_batches == 0:
        return float("inf")

    return total_loss / num_batches


def train_station(
    station_dir: Path,
    epochs: int = 10,
    batch_size: int = 0,
    learning_rate: float = 1e-3,
    patience: int = 5,
    num_workers: int = 0,
    use_amp: bool = True,
) -> None:
    station = station_dir.name
    meta_path = station_dir / "meta.json"

    if not meta_path.exists():
        print(f"[SKIP] {station} | meta.json not found")
        return

    meta = load_json(meta_path)
    cohort = meta.get("cohort", "A")
    n_features = int(meta["feature_count"])

    hidden_size, num_layers = get_model_params_by_cohort(cohort)
    bs = get_batch_size_by_cohort(cohort, batch_size)

    device = get_device()
    amp_enabled = use_amp and device.type == "cuda"

    print(f"\n=== TRAIN {station} ===")
    print(
        f"device={device} | cohort={cohort} | features={n_features} | "
        f"hidden={hidden_size} | layers={num_layers} | batch={bs}"
    )

    train_loader, val_loader, meta = build_dataloaders(
        station_dir=station_dir,
        batch_size=bs,
        num_workers=num_workers,
    )

    model = LSTMAutoencoder(
        n_features=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val_loss = float("inf")
    best_epoch = -1
    no_improve = 0
    history = []

    best_model_path = station_dir / "model.pt"
    train_log_path = station_dir / "train_log.json"

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"Train {station} Epoch {epoch+1}/{epochs}", leave=True)
        for batch in pbar:
            x = batch[0].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", enabled=amp_enabled):
                recon = model(x)
                loss = criterion(recon, x)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_batches += 1

            pbar.set_postfix(loss=f"{loss.item():.6f}")

        train_loss = train_loss / max(1, train_batches)
        val_loss = evaluate(model, val_loader, criterion, device, use_amp=use_amp)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(
            f"[{station}] Epoch {epoch+1}/{epochs} "
            f"| train_loss={train_loss:.6f} "
            f"| val_loss={val_loss:.6f}"
        )

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1024**3
            print(f"[{station}] GPU memory used: {mem:.2f} GB")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            no_improve = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "station": station,
                "cohort": cohort,
                "n_features": n_features,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
            }, best_model_path)
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[EARLY STOP] {station} | no improvement for {patience} epochs")
            break

    log_obj = {
        "station": station,
        "cohort": cohort,
        "device": str(device),
        "n_features": n_features,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "batch_size": bs,
        "learning_rate": learning_rate,
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
    }
    save_json(train_log_path, log_obj)

    print(f"[OK] saved best model -> {best_model_path}")
    print(f"[OK] saved train log  -> {train_log_path}")

    del model, train_loader, val_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    pr = project_root()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        default=str(pr / "data" / "processed" / "datasets"),
        help="Directory containing station dataset folders"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=0, help="Override batch size (0 = cohort default)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision AMP")
    parser.add_argument("--limit", type=int, default=0, help="Train only first N stations")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    station_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])

    if args.limit > 0:
        station_dirs = station_dirs[:args.limit]

    if not station_dirs:
        raise RuntimeError(f"No station dataset directories found in: {dataset_dir}")

    print(f"[INFO] stations to train: {len(station_dirs)}")
    print(f"[INFO] device: {get_device()}")

    for station_dir in station_dirs:
        train_station(
            station_dir=station _dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
            num_workers=args.num_workers,
            use_amp=not args.no_amp,
        )


if __name__ == "__main__":
    main()