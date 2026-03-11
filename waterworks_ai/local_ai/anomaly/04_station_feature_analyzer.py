import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def classify_sensor_type(sensor_name: str) -> str:
    s = str(sensor_name).upper()

    if "PIT" in s or "PRESS" in s or "압력" in sensor_name:
        return "pressure"
    if "LEVEL" in s or "LVL" in s or "수위" in sensor_name:
        return "level"
    if "FT" in s or "FLOW" in s or "유량" in sensor_name:
        return "flow"
    if "전류" in sensor_name or "AMP" in s or re_match_prefix(s, "A"):
        return "current"
    if "전압" in sensor_name or re_match_prefix(s, "V"):
        return "voltage"
    if "인버터" in sensor_name or "INV" in s:
        return "inverter"
    if "AO" in s:
        return "ao"
    return "other"


def re_match_prefix(s: str, prefix: str) -> bool:
    return s.startswith(prefix) and len(s) <= 8


def is_all_zero(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return False
    return bool((s == 0).all())


def detect_sensor_columns(df: pd.DataFrame) -> List[str]:
    exclude_cols = {"ts", "station", "source_file"}
    sensor_cols = []

    for c in df.columns:
        if c in exclude_cols:
            continue

        s = pd.to_numeric(df[c], errors="coerce")
        if int(s.notna().sum()) >= 30:
            sensor_cols.append(c)

    return sensor_cols


def sensor_priority(sensor_type: str) -> int:
    priority_map = {
        "pressure": 1,
        "level": 2,
        "current": 3,
        "voltage": 4,
        "inverter": 5,
        "ao": 6,
        "flow": 7,   # 현재는 대부분 0이니 낮게
        "other": 99,
    }
    return priority_map.get(sensor_type, 99)


def analyze_station_features(parquet_path: Path) -> Dict[str, Any]:
    df = pd.read_parquet(parquet_path)
    station_name = parquet_path.stem

    sensor_cols = detect_sensor_columns(df)

    sensor_rows = []
    selected_features = []

    for col in sensor_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        missing_rate = float(s.isna().mean())
        zero_only = is_all_zero(s)
        sensor_type = classify_sensor_type(col)

        usable = True
        exclude_reason = ""

        if zero_only:
            usable = False
            exclude_reason = "all_zero"
        elif missing_rate >= 0.1:
            usable = False
            exclude_reason = "missing_rate_ge_0.1"

        row = {
            "station": station_name,
            "sensor": col,
            "sensor_type": sensor_type,
            "missing_rate": round(missing_rate, 6),
            "all_zero": zero_only,
            "usable": usable,
            "exclude_reason": exclude_reason,
            "priority": sensor_priority(sensor_type),
        }
        sensor_rows.append(row)

        if usable:
            selected_features.append((col, sensor_type, sensor_priority(sensor_type)))

    # 우선순위 정렬
    selected_features = sorted(selected_features, key=lambda x: x[2])

    # flow는 값이 다 0인 경우가 많으므로 실제 usable한 경우만 포함됨
    feature_names = [x[0] for x in selected_features]

    summary = {
        "station": station_name,
        "sensor_count": len(sensor_cols),
        "usable_feature_count": len(feature_names),
        "selected_features": "|".join(feature_names),
    }

    return {
        "summary": summary,
        "sensor_rows": sensor_rows,
    }


def main():
    pr = project_root()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--station_dir",
        default=str(pr / "data" / "processed" / "stations"),
        help="Directory containing station parquet files"
    )
    parser.add_argument(
        "--train_list",
        default=str(pr / "data" / "reports" / "train_station_list.csv"),
        help="Train station list CSV"
    )
    parser.add_argument(
        "--out_dir",
        default=str(pr / "data" / "reports"),
        help="Output report directory"
    )
    args = parser.parse_args()

    station_dir = Path(args.station_dir)
    train_list_path = Path(args.train_list)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_list_path, encoding="utf-8-sig")
    train_stations = set(train_df["station"].astype(str).tolist())

    parquet_files = sorted(station_dir.glob("*.parquet"))
    parquet_files = [p for p in parquet_files if p.stem in train_stations]

    print(f"[INFO] Train stations to analyze: {len(parquet_files)}")

    summary_rows = []
    sensor_rows = []

    for fp in parquet_files:
        result = analyze_station_features(fp)
        summary_rows.append(result["summary"])
        sensor_rows.extend(result["sensor_rows"])

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["usable_feature_count", "sensor_count"],
        ascending=[False, False]
    )
    sensor_df = pd.DataFrame(sensor_rows).sort_values(
        ["station", "priority", "sensor"]
    )

    summary_path = out_dir / "station_feature_summary.csv"
    config_path = out_dir / "station_feature_config.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    sensor_df.to_csv(config_path, index=False, encoding="utf-8-sig")

    print(f"[OK] summary -> {summary_path}")
    print(f"[OK] config  -> {config_path}")

    print()
    print("=== TOP 10 STATIONS BY FEATURE COUNT ===")
    print(summary_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()