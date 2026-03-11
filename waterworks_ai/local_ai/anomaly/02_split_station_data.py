import argparse
from pathlib import Path
from collections import defaultdict
import re

import pandas as pd
from tqdm import tqdm


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sanitize_station_name(name: str) -> str:
    name = str(name).strip()
    name = name.replace("/", "_")
    name = name.replace("\\", "_")
    name = name.replace(" ", "")
    return name


def detect_encoding(csv_path: Path) -> str:
    for enc in ["cp949", "utf-8-sig", "utf-8", "euc-kr"]:
        try:
            pd.read_csv(csv_path, nrows=3, encoding=enc)
            return enc
        except Exception:
            continue
    raise RuntimeError(f"Cannot detect encoding for {csv_path}")


def find_time_col(columns) -> str:
    candidates = ["시각", "time", "timestamp", "datetime", "date", "ts"]
    for cand in candidates:
        if cand in columns:
            return cand
    return columns[0]


def split_column_name(col: str):
    """
    실제 CSV 구조를 기준으로 station / sensor 분리

    처리 예시:
    - 가곡고지.AI.PIT_102 -> station=가곡고지, sensor=PIT_102
    - 주암송광.가압장.AI.A101 -> station=주암송광.가압장, sensor=A101
    - 주암송광.배수지.AI.FT302_순시 -> station=주암송광.배수지, sensor=FT302_순시
    - 청소가압장.압력.흡입.LEVEL -> station=청소가압장, sensor=압력_흡입_LEVEL
    - 행복가압장.전력감시.V -> station=행복가압장, sensor=전력감시_V
    """

    col = str(col).strip()
    if "." not in col:
        return None, None

    # 1) AI / AO / DI 패턴 우선 처리
    for marker in [".AI.", ".AO.", ".DI."]:
        if marker in col:
            left, right = col.split(marker, 1)
            station = left.strip()
            sensor = right.strip().replace(".", "_")
            if station and sensor:
                return station, sensor

    # 2) 일반 패턴 처리
    # station은 첫 번째 토큰, 나머지는 sensor로 합침
    parts = col.split(".")
    if len(parts) >= 2:
        station = parts[0].strip()
        sensor = "_".join([p.strip() for p in parts[1:] if p.strip()])
        if station and sensor:
            return station, sensor

    return None, None


def make_unique_columns(columns):
    """
    중복 컬럼명을 pandas concat 가능하도록 유니크하게 변환
    예:
      PIT_101, PIT_101 -> PIT_101, PIT_101__dup1
    """
    seen = {}
    out = []

    for c in columns:
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__dup{seen[c]}")

    return out


def build_station_column_map(columns, time_col):
    station_map = defaultdict(list)

    for col in columns:
        if col == time_col:
            continue

        station, sensor = split_column_name(col)
        if station is None:
            continue

        station_map[station].append(col)

    return dict(station_map)


def process_one_csv(csv_path: Path, out_dir: Path):
    encoding = detect_encoding(csv_path)
    print(f"[INFO] Reading {csv_path.name} (encoding={encoding})")

    df = pd.read_csv(csv_path, encoding=encoding)
    time_col = find_time_col(df.columns)

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    station_map = build_station_column_map(df.columns, time_col)
    print(f"[INFO] Found stations in {csv_path.name}: {len(station_map)}")

    for station, cols in tqdm(station_map.items(), desc=f"Split {csv_path.name}"):
        station_name = sanitize_station_name(station)

        # 원본에서 해당 station 관련 컬럼만 추출
        station_df = df[[time_col] + cols].copy()

        # sensor 이름으로 rename
        rename_map = {}
        new_cols = ["ts"]

        for c in cols:
            _, sensor = split_column_name(c)
            if sensor is None:
                sensor = c
            rename_map[c] = sensor
            new_cols.append(sensor)

        station_df = station_df.rename(columns={time_col: "ts", **rename_map})

        # 중복 sensor 이름 처리
        station_df.columns = make_unique_columns(station_df.columns)

        # station 메타 추가
        station_df["station"] = station_name
        station_df["source_file"] = csv_path.name

        # ts 기준 정렬
        station_df = station_df.sort_values("ts").reset_index(drop=True)

        out_path = out_dir / f"{station_name}.parquet"

        if out_path.exists():
            old_df = pd.read_parquet(out_path)

            # 컬럼 중복 있으면 정리
            old_df = old_df.loc[:, ~old_df.columns.duplicated()]
            station_df = station_df.loc[:, ~station_df.columns.duplicated()]

            merged = pd.concat(
                [old_df, station_df],
                ignore_index=True,
                sort=False
            )

            merged = merged.loc[:, ~merged.columns.duplicated()]
            merged = merged.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
            merged.to_parquet(out_path, index=False)
        else:
            station_df.to_parquet(out_path, index=False)


def main():
    pr = project_root()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir",
        default=str(pr / "data" / "raw" / "sensor_csv"),
        help="Directory containing monthly sensor CSV files"
    )
    parser.add_argument(
        "--out_dir",
        default=str(pr / "data" / "processed" / "stations"),
        help="Directory to save station parquet files"
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="CSV filename glob pattern"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N files (0 = all)"
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(in_dir.glob(args.pattern))
    if args.limit and args.limit > 0:
        csv_files = csv_files[:args.limit]

    if not csv_files:
        raise RuntimeError(f"No CSV files found in {in_dir}")

    print(f"[INFO] Total CSV files: {len(csv_files)}")
    print("[INFO] Files:", [p.name for p in csv_files[:5]])

    for csv_path in csv_files:
        process_one_csv(csv_path, out_dir)

    print(f"[DONE] Station parquet files saved to: {out_dir}")


if __name__ == "__main__":
    main()