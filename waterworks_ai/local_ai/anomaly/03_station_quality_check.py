import argparse
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def detect_sensor_columns(df: pd.DataFrame) -> List[str]:
    """
    parquet에서 실제 센서 컬럼만 선택
    제외:
    - ts
    - station
    - source_file
    """
    exclude_cols = {"ts", "station", "source_file"}
    sensor_cols = []

    for c in df.columns:
        if c in exclude_cols:
            continue

        s = pd.to_numeric(df[c], errors="coerce")
        valid_points = int(s.notna().sum())

        # 최소한 값이 있어야 센서로 인정
        if valid_points >= 30:
            sensor_cols.append(c)

    return sensor_cols


def analyze_station_file(parquet_path: Path) -> Dict[str, Any]:
    df = pd.read_parquet(parquet_path)

    station_name = parquet_path.stem
    row_count = len(df)

    if "ts" in df.columns:
        ts_valid = pd.to_datetime(df["ts"], errors="coerce").notna().sum()
    else:
        ts_valid = 0

    sensor_cols = detect_sensor_columns(df)

    sensor_rows = []
    for col in sensor_cols:
        s = pd.to_numeric(df[col], errors="coerce")

        valid_points = int(s.notna().sum())
        missing_rate = float(s.isna().mean())

        sensor_rows.append({
            "station": station_name,
            "sensor": col,
            "valid_points": valid_points,
            "missing_rate": round(missing_rate, 6),
        })

    if sensor_rows:
        sensor_df = pd.DataFrame(sensor_rows)
        avg_missing_rate = float(sensor_df["missing_rate"].mean())
        usable_sensor_count = int((sensor_df["missing_rate"] < 0.1).sum())
    else:
        avg_missing_rate = 1.0
        usable_sensor_count = 0

    # -------------------------
    # 제외 규칙
    # -------------------------
    exclude_reasons = []

    # 실제 가압장/관측소가 아닐 가능성이 높은 이름
    suspicious_keywords = ["가상태그", "구역유량"]
    for kw in suspicious_keywords:
        if kw in station_name:
            exclude_reasons.append(f"name_contains:{kw}")

    if row_count == 0:
        exclude_reasons.append("empty_rows")

    if ts_valid == 0:
        exclude_reasons.append("invalid_ts")

    if len(sensor_cols) < 3:
        exclude_reasons.append("sensor_count_lt_3")

    if avg_missing_rate >= 0.1:
        exclude_reasons.append("avg_missing_rate_ge_0.1")

    if usable_sensor_count < 2:
        exclude_reasons.append("usable_sensor_count_lt_2")

    usable = len(exclude_reasons) == 0

    summary = {
        "station": station_name,
        "row_count": row_count,
        "ts_valid_count": ts_valid,
        "sensor_count": len(sensor_cols),
        "usable_sensor_count": usable_sensor_count,
        "avg_missing_rate": round(avg_missing_rate, 6),
        "usable": usable,
        "exclude_reason": " | ".join(exclude_reasons) if exclude_reasons else "",
    }

    return {
        "summary": summary,
        "sensor_rows": sensor_rows,
    }


def main():
    pr = project_root()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir",
        default=str(pr / "data" / "processed" / "stations"),
        help="Directory containing station parquet files"
    )
    parser.add_argument(
        "--out_dir",
        default=str(pr / "data" / "reports"),
        help="Directory to save quality check reports"
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(in_dir.glob("*.parquet"))

    if not parquet_files:
        raise RuntimeError(f"No parquet files found in: {in_dir}")

    print(f"[INFO] Station parquet files found: {len(parquet_files)}")

    summary_rows = []
    sensor_rows = []

    for fp in parquet_files:
        result = analyze_station_file(fp)
        summary_rows.append(result["summary"])
        sensor_rows.extend(result["sensor_rows"])

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["usable", "avg_missing_rate", "sensor_count"],
        ascending=[False, True, False]
    )

    sensor_df = pd.DataFrame(sensor_rows)

    train_df = summary_df[summary_df["usable"] == True].copy()
    excluded_df = summary_df[summary_df["usable"] == False].copy()

    summary_path = out_dir / "station_quality_summary.csv"
    sensor_path = out_dir / "station_sensor_quality.csv"
    train_path = out_dir / "train_station_list.csv"
    excluded_path = out_dir / "excluded_station_list.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    sensor_df.to_csv(sensor_path, index=False, encoding="utf-8-sig")
    train_df[["station"]].to_csv(train_path, index=False, encoding="utf-8-sig")
    excluded_df.to_csv(excluded_path, index=False, encoding="utf-8-sig")

    print(f"[OK] summary -> {summary_path}")
    print(f"[OK] sensor  -> {sensor_path}")
    print(f"[OK] train   -> {train_path}")
    print(f"[OK] exclude -> {excluded_path}")

    print()
    print("=== RESULT ===")
    print(f"Total stations   : {len(summary_df)}")
    print(f"Train stations   : {len(train_df)}")
    print(f"Excluded stations: {len(excluded_df)}")

    if len(train_df) > 0:
        print("\n[Top 10 train candidates]")
        print(train_df.head(10).to_string(index=False))

    if len(excluded_df) > 0:
        print("\n[Top 10 excluded stations]")
        print(excluded_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()