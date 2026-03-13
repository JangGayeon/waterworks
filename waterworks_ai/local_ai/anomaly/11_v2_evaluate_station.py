import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt

from model import LSTMAutoencoder


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_for_station(station_dir: Path, device: torch.device) -> Tuple[LSTMAutoencoder, Dict[str, Any]]:
    ckpt = torch.load(station_dir / "model.pt2", map_location=device, weights_only=False)

    n_features = int(ckpt["n_features"])
    hidden_size = int(ckpt["hidden_size"])
    num_layers = int(ckpt["num_layers"])

    model = LSTMAutoencoder(
        n_features=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt


def reconstruct(model, X: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    preds = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            xb = torch.tensor(batch, dtype=torch.float32, device=device)
            recon = model(xb).detach().cpu().numpy()
            preds.append(recon)

    if not preds:
        return np.empty_like(X)

    return np.concatenate(preds, axis=0)


def inverse_transform_sequences(X_seq: np.ndarray, scaler) -> np.ndarray:
    n, t, f = X_seq.shape
    flat = X_seq.reshape(-1, f)
    inv = scaler.inverse_transform(flat)
    return inv.reshape(n, t, f)


def threshold_candidates(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std())
    return {
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean_plus_3std": float(mean + 3 * std),
        "mean": mean,
        "std": std,
        "max": float(arr.max()),
    }


def normalize_weights(feature_names: List[str], sensor_weights: Dict[str, float]) -> np.ndarray:
    weights = []
    for feat in feature_names:
        weights.append(float(sensor_weights.get(feat, 1.0)))

    weights = np.array(weights, dtype=float)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    return weights


def get_critical_feature_indices(feature_names: List[str], feature_types: Dict[str, str]) -> List[int]:
    critical_types = {"pressure", "level"}
    idx = []
    for i, feat in enumerate(feature_names):
        if feature_types.get(feat, "unknown") in critical_types:
            idx.append(i)
    return idx


def compute_scores(
    actual: np.ndarray,
    recon: np.ndarray,
    feature_names: List[str],
    feature_types: Dict[str, str],
    sensor_weights: Dict[str, float],
) -> Dict[str, Any]:
    abs_err = np.abs(actual - recon)
    sq_err = (actual - recon) ** 2

    # 기본 점수
    seq_mae = abs_err.mean(axis=(1, 2))
    seq_mse = sq_err.mean(axis=(1, 2))
    seq_rmse = np.sqrt(seq_mse)

    # feature별 평균
    feature_mae = abs_err.mean(axis=(0, 1))
    feature_rmse = np.sqrt(sq_err.mean(axis=(0, 1)))

    # weighted score
    weights = normalize_weights(feature_names, sensor_weights)
    weighted_seq_mae = (abs_err * weights.reshape(1, 1, -1)).sum(axis=2).mean(axis=1)

    # critical sensors max (pressure/level)
    critical_idx = get_critical_feature_indices(feature_names, feature_types)
    if len(critical_idx) > 0:
        critical_abs = abs_err[:, :, critical_idx]
        critical_sensor_max = critical_abs.max(axis=(1, 2))
    else:
        critical_sensor_max = np.zeros(len(seq_mae), dtype=float)

    # 센서 타입별 score
    type_group_scores: Dict[str, np.ndarray] = {}
    for sensor_type in ["pressure", "level", "flow", "current", "voltage", "inverter", "zt", "ao", "state"]:
        idx = [i for i, feat in enumerate(feature_names) if feature_types.get(feat, "unknown") == sensor_type]
        if len(idx) > 0:
            type_group_scores[sensor_type] = abs_err[:, :, idx].mean(axis=(1, 2))
        else:
            type_group_scores[sensor_type] = np.zeros(len(seq_mae), dtype=float)

    return {
        "abs_err": abs_err,
        "sq_err": sq_err,
        "seq_mae": seq_mae,
        "seq_rmse": seq_rmse,
        "weighted_seq_mae": weighted_seq_mae,
        "critical_sensor_max": critical_sensor_max,
        "feature_mae": feature_mae,
        "feature_rmse": feature_rmse,
        "type_group_scores": type_group_scores,
        "weights": weights,
    }


def make_output_dir(station_dir: Path) -> Path:
    out_dir = station_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_actual_vs_recon(
    actual: np.ndarray,
    recon: np.ndarray,
    feature_names: List[str],
    out_path: Path,
    sample_index: int = 0,
    max_features: int = 4,
):
    seq_actual = actual[sample_index]
    seq_recon = recon[sample_index]

    num_features = min(len(feature_names), max_features)

    plt.figure(figsize=(14, 3 * num_features))
    for i in range(num_features):
        plt.subplot(num_features, 1, i + 1)
        plt.plot(seq_actual[:, i], label="actual")
        plt.plot(seq_recon[:, i], label="recon")
        plt.title(f"{feature_names[i]} | sample={sample_index}")
        plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_series(arr: np.ndarray, title: str, ylabel: str, out_path: Path):
    plt.figure(figsize=(14, 4))
    plt.plot(arr)
    plt.title(title)
    plt.xlabel("Sequence index")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_hist(arr: np.ndarray, title: str, xlabel: str, out_path: Path):
    plt.figure(figsize=(8, 4))
    plt.hist(arr, bins=100)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    pr = project_root()

    parser = argparse.ArgumentParser()
    parser.add_argument("--station", required=True, help="Station name, e.g. 가곡고지")
    parser.add_argument(
        "--dataset_dir",
        default=str(pr / "data" / "processed" / "datasets"),
        help="Directory containing station dataset folders"
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Validation sample index to visualize"
    )
    parser.add_argument(
        "--max_features_plot",
        type=int,
        default=4,
        help="Max number of features to draw in actual_vs_recon plot"
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    station_dir = dataset_dir / args.station

    if not station_dir.exists():
        raise RuntimeError(f"Station dataset directory not found: {station_dir}")

    meta = load_json(station_dir / "meta.json")
    feat_info = load_json(station_dir / "feature_columns.json")

    feature_names = feat_info["features"]
    feature_types = feat_info.get("feature_types", {})
    sensor_weights = feat_info.get("sensor_weights", {})

    X_val = np.load(station_dir / "X_val.npy")

    with open(station_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    device = get_device()
    model, model_ckpt = load_model_for_station(station_dir, device)

    print(f"[INFO] station={args.station}")
    print(f"[INFO] device={device}")
    print(f"[INFO] X_val shape={X_val.shape}")

    X_recon = reconstruct(model, X_val, device=device, batch_size=1024)

    actual_inv = inverse_transform_sequences(X_val, scaler)
    recon_inv = inverse_transform_sequences(X_recon, scaler)

    scores = compute_scores(
        actual=actual_inv,
        recon=recon_inv,
        feature_names=feature_names,
        feature_types=feature_types,
        sensor_weights=sensor_weights,
    )

    seq_mae_th = threshold_candidates(scores["seq_mae"])
    weighted_th = threshold_candidates(scores["weighted_seq_mae"])
    critical_th = threshold_candidates(scores["critical_sensor_max"])

    out_dir = make_output_dir(station_dir)

    sample_index = max(0, min(args.sample_index, len(actual_inv) - 1))

    plot_actual_vs_recon(
        actual=actual_inv,
        recon=recon_inv,
        feature_names=feature_names,
        out_path=out_dir / "actual_vs_recon.png",
        sample_index=sample_index,
        max_features=args.max_features_plot,
    )

    plot_series(
        scores["seq_mae"],
        "Sequence MAE over validation windows",
        "MAE",
        out_dir / "residual_timeseries.png"
    )

    plot_hist(
        scores["seq_mae"],
        "Residual MAE Histogram",
        "MAE",
        out_dir / "residual_hist.png"
    )

    plot_series(
        scores["weighted_seq_mae"],
        "Weighted Sequence MAE over validation windows",
        "Weighted MAE",
        out_dir / "weighted_residual_timeseries.png"
    )

    plot_series(
        scores["critical_sensor_max"],
        "Critical Sensor Max Residual over validation windows",
        "Critical Max Residual",
        out_dir / "critical_sensor_timeseries.png"
    )

    feature_mae = {
        feature_names[i]: float(scores["feature_mae"][i])
        for i in range(len(feature_names))
    }
    feature_rmse = {
        feature_names[i]: float(scores["feature_rmse"][i])
        for i in range(len(feature_names))
    }

    type_group_thresholds = {
        sensor_type: threshold_candidates(arr)
        for sensor_type, arr in scores["type_group_scores"].items()
    }

    summary = {
        "station": args.station,
        "device": str(device),
        "cohort": meta.get("cohort"),
        "feature_count": meta.get("feature_count"),
        "features": feature_names,
        "feature_types": feature_types,
        "sensor_weights": sensor_weights,
        "normalized_weights": {
            feature_names[i]: float(scores["weights"][i])
            for i in range(len(feature_names))
        },
        "model_best_epoch": model_ckpt.get("best_epoch"),
        "model_best_val_loss": model_ckpt.get("best_val_loss"),
        "val_sequence_count": int(len(X_val)),
        "sample_index_visualized": sample_index,
        "threshold_candidates": {
            "seq_mae": seq_mae_th,
            "weighted_seq_mae": weighted_th,
            "critical_sensor_max": critical_th,
            "type_group_scores": type_group_thresholds,
        },
        "feature_mae": feature_mae,
        "feature_rmse": feature_rmse,
    }

    save_json(out_dir / "evaluation_summary.json", summary)

    print(f"[OK] saved -> {out_dir / 'actual_vs_recon.png'}")
    print(f"[OK] saved -> {out_dir / 'residual_timeseries.png'}")
    print(f"[OK] saved -> {out_dir / 'residual_hist.png'}")
    print(f"[OK] saved -> {out_dir / 'weighted_residual_timeseries.png'}")
    print(f"[OK] saved -> {out_dir / 'critical_sensor_timeseries.png'}")
    print(f"[OK] saved -> {out_dir / 'evaluation_summary.json'}")

    print("\n=== THRESHOLD CANDIDATES ===")
    print("[seq_mae]")
    for k, v in seq_mae_th.items():
        print(f"  {k}: {v:.6f}")

    print("[weighted_seq_mae]")
    for k, v in weighted_th.items():
        print(f"  {k}: {v:.6f}")

    print("[critical_sensor_max]")
    for k, v in critical_th.items():
        print(f"  {k}: {v:.6f}")

    print("\n=== FEATURE MAE ===")
    for k, v in feature_mae.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()