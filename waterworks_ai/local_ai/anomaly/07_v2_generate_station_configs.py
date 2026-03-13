import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_feature_list(feature_str: str) -> List[str]:
    if pd.isna(feature_str) or str(feature_str).strip() == "":
        return []
    return [x.strip() for x in str(feature_str).split("|") if x.strip()]


def infer_cohort(feature_count: int) -> str:
    if feature_count >= 5:
        return "A"
    if 3 <= feature_count <= 4:
        return "B"
    return "C"


def default_weight_by_type(sensor_type: str) -> float:
    sensor_type = str(sensor_type)

    if sensor_type == "pressure":
        return 1.0
    if sensor_type == "level":
        return 1.0
    if sensor_type == "flow":
        return 0.8
    if sensor_type == "current":
        return 0.7
    if sensor_type == "voltage":
        return 0.6
    if sensor_type == "inverter":
        return 0.7
    if sensor_type == "zt":
        return 0.6
    if sensor_type == "ao":
        return 0.5
    if sensor_type == "state":
        return 0.3

    return 0.4


def main():
    pr = project_root()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_csv",
        default=str(pr / "data" / "reports" / "station_model_config.csv"),
        help="Path to station_model_config.csv"
    )
    parser.add_argument(
        "--detail_csv",
        default=str(pr / "data" / "reports" / "station_model_config_detail.csv"),
        help="Path to station_model_config_detail.csv"
    )
    parser.add_argument(
        "--out_dir",
        default=str(pr / "configs" / "stations"),
        help="Directory to save station json configs"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=60,
        help="Sequence length"
    )
    parser.add_argument(
        "--sampling_minutes",
        type=int,
        default=1,
        help="Sampling interval in minutes"
    )
    args = parser.parse_args()

    config_csv = Path(args.config_csv)
    detail_csv = Path(args.detail_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_df = pd.read_csv(config_csv, encoding="utf-8-sig")
    detail_df = pd.read_csv(detail_csv, encoding="utf-8-sig")

    if "station" not in config_df.columns:
        raise RuntimeError("station_model_config.csv must contain 'station' column")

    created = 0

    for _, row in config_df.iterrows():
        station = str(row["station"]).strip()
        feature_count = int(row.get("selected_feature_count", 0))
        features = parse_feature_list(row.get("selected_features", ""))
        cohort = infer_cohort(feature_count)

        sub = detail_df[detail_df["station"].astype(str) == station].copy()

        feature_types: Dict[str, str] = {}
        for feat in features:
            matched = sub[sub["sensor"].astype(str) == feat]
            if len(matched) > 0 and "sensor_type" in matched.columns:
                feature_types[feat] = str(matched.iloc[0]["sensor_type"])
            else:
                feature_types[feat] = "unknown"

        sensor_weights: Dict[str, float] = {}
        for feat in features:
            sensor_weights[feat] = default_weight_by_type(feature_types.get(feat, "unknown"))

        obj: Dict[str, Any] = {
            "station": station,
            "features": features,
            "feature_types": feature_types,
            "sensor_weights": sensor_weights,
            "sequence_length": int(args.seq_len),
            "sampling_minutes": int(args.sampling_minutes),
            "cohort": cohort,
            "main_sensor_types": ["pressure", "level"],
            "secondary_sensor_types": ["flow", "current", "voltage", "inverter", "zt", "ao"],
            "score_policy": {
                "primary_focus": ["pressure", "level"],
                "use_weighted_residual": True,
                "use_critical_sensor_max": True,
                "deadband_required": True
            }
        }

        out_path = out_dir / f"{station}.json"
        out_path.write_text(
            json.dumps(obj, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        created += 1

    print(f"[OK] Station JSON configs created: {created}")
    print(f"[OK] Output dir: {out_dir}")


if __name__ == "__main__":
    main()