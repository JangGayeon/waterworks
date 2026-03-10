import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_datetime_and_sort(df: pd.DataFrame, time_col: str = "ts") -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    return out


def fill_missing(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    out = df.copy()

    for c in feature_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # 기본 결측 처리
    out[feature_cols] = out[feature_cols].ffill().bfill()
    out[feature_cols] = out[feature_cols].interpolate(method="linear", limit_direction="both")

    return out


def drop_bad_rows(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=feature_cols, how="all")
    return out.reset_index(drop=True)


def build_sequences(arr: np.ndarray, seq_len: int) -> np.ndarray:
    n = len(arr)
    if n < seq_len:
        return np.empty((0, seq_len, arr.shape[1]), dtype=np.float32)

    windows = []
    for i in range(n - seq_len + 1):
        windows.append(arr[i:i + seq_len])

    return np.array(windows, dtype=np.float32)


def train_val_split(X: np.ndarray, val_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) == 0:
        return X, X

    split_idx = int(len(X) * (1 - val_ratio))
    split_idx = max(1, split_idx)

    X_train = X[:split_idx]
    X_val = X[split_idx:]

    return X_train, X_val


def process_station(
    station: str,
    config: Dict[str, Any],
    station_parquet_path: Path,
    out_station_dir: Path,
    val_ratio: float = 0.2,
) -> Dict[str, Any]:
    df = pd.read_parquet(station_parquet_path)
    df = ensure_datetime_and_sort(df, time_col="ts")

    feature_cols = config.get("features", [])
    seq_len = int(config.get("sequence_length", 60))
    sampling_minutes = int(config.get("sampling_minutes", 1))
    cohort = str(config.get("cohort", "unknown"))

    # parquet에 실제 존재하는 feature만 사용
    available_features = [c for c in feature_cols if c in df.columns]

    if len(available_features) == 0:
        return {
            "station": station,
            "status": "skip",
            "reason": "no_available_features",
            "cohort": cohort,
            "feature_count": 0,
            "row_count": 0,
            "sequence_count": 0,
        }

    work_df = df[["ts"] + available_features].copy()
    work_df = drop_bad_rows(work_df, available_features)
    work_df = fill_missing(work_df, available_features)

    X_df = work_df[available_features].copy()

    for c in available_features:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")

    X_df = X_df.dropna().reset_index(drop=True)

    if len(X_df) < seq_len:
        return {
            "station": station,
            "status": "skip",
            "reason": "not_enough_rows",
            "cohort": cohort,
            "feature_count": len(available_features),
            "row_count": len(X_df),
            "sequence_count": 0,
        }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)

    X_seq = build_sequences(X_scaled, seq_len=seq_len)
    X_train, X_val = train_val_split(X_seq, val_ratio=val_ratio)

    out_station_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_station_dir / "X_train.npy", X_train)
    np.save(out_station_dir / "X_val.npy", X_val)

    with open(out_station_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    save_json(
        out_station_dir / "feature_columns.json",
        {
            "station": station,
            "features": available_features,
            "feature_types": config.get("feature_types", {}),
            "sensor_weights": config.get("sensor_weights", {}),
        }
    )

    save_json(
        out_station_dir / "meta.json",
        {
            "station": station,
            "cohort": cohort,
            "feature_count": len(available_features),
            "features": available_features,
            "row_count": int(len(X_df)),
            "sequence_length": seq_len,
            "sampling_minutes": sampling_minutes,
            "train_sequence_count": int(len(X_train)),
            "val_sequence_count": int(len(X_val)),
            "val_ratio": val_ratio,
        }
    )

    return {
        "station": station,
        "status": "ok",
        "reason": "",
        "cohort": cohort,
        "feature_count": len(available_features),
        "row_count": len(X_df),
        "sequence_count": len(X_seq),
        "train_sequence_count": len(X_train),
        "val_sequence_count": len(X_val),
    }


def main():
    pr = project_root()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_dir",
        default=str(pr / "configs" / "stations"),
        help="Directory containing station json configs"
    )
    parser.add_argument(
        "--station_dir",
        default=str(pr / "data" / "processed" / "stations"),
        help="Directory containing station parquet files"
    )
    parser.add_argument(
        "--out_dir",
        default=str(pr / "data" / "processed" / "datasets"),
        help="Output directory for station datasets"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation ratio"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N configs (0=all)"
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    station_dir = Path(args.station_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(config_dir.glob("*.json"))
    if args.limit > 0:
        json_files = json_files[:args.limit]

    if not json_files:
        raise RuntimeError(f"No station config json files found in: {config_dir}")

    print(f"[INFO] Station configs to process: {len(json_files)}")

    results = []

    for json_path in json_files:
        config = load_json(json_path)
        station = str(config.get("station", json_path.stem)).strip()

        station_parquet_path = station_dir / f"{station}.parquet"
        out_station_dir = out_dir / station

        if not station_parquet_path.exists():
            results.append({
                "station": station,
                "status": "skip",
                "reason": "parquet_not_found",
                "cohort": config.get("cohort", "unknown"),
                "feature_count": len(config.get("features", [])),
                "row_count": 0,
                "sequence_count": 0,
            })
            print(f"[SKIP] {station} | parquet_not_found")
            continue

        result = process_station(
            station=station,
            config=config,
            station_parquet_path=station_parquet_path,
            out_station_dir=out_station_dir,
            val_ratio=args.val_ratio,
        )
        results.append(result)
        print(
            f"[{result['status'].upper()}] {station} | cohort={result['cohort']} | "
            f"features={result['feature_count']} | rows={result['row_count']} | seq={result['sequence_count']}"
        )

    result_df = pd.DataFrame(results)
    result_path = out_dir / "dataset_build_summary.csv"
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")

    print(f"\n[OK] summary -> {result_path}")
    print("\n=== RESULT ===")
    print(result_df["status"].value_counts(dropna=False))
    if "cohort" in result_df.columns:
        print("\n[COHORT SUMMARY]")
        print(result_df.groupby(["cohort", "status"]).size())


if __name__ == "__main__":
    main()