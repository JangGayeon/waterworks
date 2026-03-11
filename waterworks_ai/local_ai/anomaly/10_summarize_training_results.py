import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_one_station(station_dir: Path) -> Dict[str, Any]:
    station = station_dir.name
    log_path = station_dir / "train_log.json"
    meta_path = station_dir / "meta.json"
    model_path = station_dir / "model.pt"

    row: Dict[str, Any] = {
        "station": station,
        "status": "missing_log",
        "cohort": None,
        "feature_count": None,
        "sequence_length": None,
        "row_count": None,
        "train_sequence_count": None,
        "val_sequence_count": None,
        "device": None,
        "hidden_size": None,
        "num_layers": None,
        "batch_size": None,
        "learning_rate": None,
        "epochs_requested": None,
        "epochs_ran": None,
        "best_epoch": None,
        "best_val_loss": None,
        "last_train_loss": None,
        "last_val_loss": None,
        "model_exists": model_path.exists(),
    }

    if meta_path.exists():
        meta = load_json(meta_path)
        row["cohort"] = meta.get("cohort")
        row["feature_count"] = meta.get("feature_count")
        row["sequence_length"] = meta.get("sequence_length")
        row["row_count"] = meta.get("row_count")
        row["train_sequence_count"] = meta.get("train_sequence_count")
        row["val_sequence_count"] = meta.get("val_sequence_count")

    if not log_path.exists():
        return row

    log = load_json(log_path)
    history: List[Dict[str, Any]] = log.get("history", [])

    row["status"] = "ok"
    row["device"] = log.get("device")
    row["hidden_size"] = log.get("hidden_size")
    row["num_layers"] = log.get("num_layers")
    row["batch_size"] = log.get("batch_size")
    row["learning_rate"] = log.get("learning_rate")
    row["epochs_requested"] = log.get("epochs_requested")
    row["epochs_ran"] = len(history)
    row["best_epoch"] = log.get("best_epoch")
    row["best_val_loss"] = log.get("best_val_loss")

    if history:
        row["last_train_loss"] = history[-1].get("train_loss")
        row["last_val_loss"] = history[-1].get("val_loss")

    return row


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def judge(row):
        if row["status"] != "ok":
            return "missing_log"
        if pd.isna(row["best_val_loss"]):
            return "unknown"
        if row["best_val_loss"] <= 0.001:
            return "excellent"
        if row["best_val_loss"] <= 0.003:
            return "good"
        if row["best_val_loss"] <= 0.01:
            return "ok"
        return "review"

    out["quality_flag"] = out.apply(judge, axis=1)

    def overfit_gap(row):
        if pd.isna(row["last_train_loss"]) or pd.isna(row["last_val_loss"]):
            return None
        return row["last_val_loss"] - row["last_train_loss"]

    out["val_train_gap"] = out.apply(overfit_gap, axis=1)

    return out


def main():
    pr = project_root()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        default=str(pr / "data" / "processed" / "datasets"),
        help="Directory containing station dataset folders"
    )
    parser.add_argument(
        "--out_dir",
        default=str(pr / "data" / "reports"),
        help="Output directory"
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    station_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])

    if not station_dirs:
        raise RuntimeError(f"No station dataset directories found in: {dataset_dir}")

    rows = [summarize_one_station(st) for st in station_dirs]
    df = pd.DataFrame(rows)
    df = add_quality_flags(df)

    df = df.sort_values(
        ["status", "quality_flag", "best_val_loss", "station"],
        ascending=[True, True, True, True]
    )

    summary_path = out_dir / "training_summary.csv"
    review_path = out_dir / "training_review_list.csv"

    df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    review_df = df[
        (df["status"] != "ok") |
        (df["quality_flag"].isin(["review", "unknown"]))
    ].copy()
    review_df.to_csv(review_path, index=False, encoding="utf-8-sig")

    print(f"[OK] summary -> {summary_path}")
    print(f"[OK] review  -> {review_path}")

    print("\n=== RESULT ===")
    print(df["quality_flag"].value_counts(dropna=False))

    print("\n=== COHORT x QUALITY ===")
    print(pd.crosstab(df["cohort"], df["quality_flag"], dropna=False))

    print("\n=== TOP 10 BEST MODELS ===")
    best_df = df[df["status"] == "ok"].sort_values(["best_val_loss", "station"]).head(10)
    print(
        best_df[
            ["station", "cohort", "feature_count", "best_epoch", "best_val_loss", "quality_flag"]
        ].to_string(index=False)
    )

    if len(review_df) > 0:
        print("\n=== REVIEW NEEDED ===")
        print(
            review_df[
                ["station", "status", "cohort", "feature_count", "best_val_loss", "quality_flag"]
            ].head(20).to_string(index=False)
        )


if __name__ == "__main__":
    main()