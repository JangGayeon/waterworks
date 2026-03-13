import argparse
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def classify_sensor_type(sensor_name: str) -> str:
    s = str(sensor_name).upper()
    raw = str(sensor_name)

    # pressure
    if any(k in raw for k in ["압력", "유입압력", "유출압력", "토출압력", "흡입압력", "토출", "흡입"]):
        return "pressure"
    if "PIT" in s or s.startswith("PT"):
        return "pressure"

    # level
    if any(k in raw for k in ["수위", "배수지수위", "저수지수위"]):
        return "level"
    if s.startswith("LT") or "LEVEL" in s:
        return "level"

    # flow
    if any(k in raw for k in ["유량", "순시유량", "누적유량"]):
        return "flow"
    if s.startswith("FT") or "FLOW" in s:
        return "flow"

    # current
    if any(k in raw for k in ["전류", "모터전류"]):
        return "current"
    if "AMP" in s or (s.startswith("A") and len(s) <= 10):
        return "current"

    # voltage
    if any(k in raw for k in ["전압", "모터전압"]):
        return "voltage"
    if s.startswith("V") and len(s) <= 10:
        return "voltage"

    # inverter
    if "인버터" in raw or "INV" in s or "RPM" in s or "주파수" in raw:
        return "inverter"

    # zt
    if s.startswith("ZT"):
        return "zt"

    # ao
    if "AO" in s:
        return "ao"

    # LS/HH/HI/LL/LO 같은 알람/상태 센서
    if s.startswith("LS") or any(k in s for k in ["HH", "HI", "LL", "LO"]):
        return "state"

    return "other"


def sensor_type_priority(sensor_type: str) -> int:
    """
    낮을수록 우선순위 높음
    """
    priority_map = {
        "pressure": 1,
        "level": 2,
        "flow": 3,
        "current": 4,
        "voltage": 5,
        "inverter": 6,
        "zt": 7,
        "ao": 8,
        "state": 9,
        "other": 99,
    }
    return priority_map.get(sensor_type, 99)


def sensor_type_limit(sensor_type: str) -> int:
    """
    압력/수위를 최우선으로 두는 제한
    """
    limit_map = {
        "pressure": 3,
        "level": 2,
        "flow": 1,
        "current": 2,
        "voltage": 1,
        "inverter": 2,
        "zt": 1,
        "ao": 1,
        "state": 2,
        "other": 0,
    }
    return limit_map.get(sensor_type, 0)


def load_feature_detail(detail_path: Path) -> pd.DataFrame:
    df = pd.read_csv(detail_path, encoding="utf-8-sig")

    # 재분류
    df["sensor_type"] = df["sensor"].astype(str).apply(classify_sensor_type)
    df["priority"] = df["sensor_type"].apply(sensor_type_priority)

    if "usable" not in df.columns:
        df["usable"] = True
    if "missing_rate" not in df.columns:
        df["missing_rate"] = 0.0
    if "std" not in df.columns:
        df["std"] = 0.0
    if "all_zero" not in df.columns:
        df["all_zero"] = False
    if "exclude_reason" not in df.columns:
        df["exclude_reason"] = ""

    return df


def select_features_for_station(
    station_df: pd.DataFrame,
    min_features: int = 4
) -> List[Dict[str, Any]]:
    """
    선택 원칙
    1. usable 센서만 사용
    2. 압력/수위 최우선
    3. 전류/전압/인버터/유량은 보조
    4. 상태(LS/HH/HI/LL/LO)는 보조 이벤트용으로만 제한적으로 포함
    """

    usable = station_df[
        (station_df["usable"] == True) &
        (station_df["all_zero"] == False)
    ].copy()

    selected_rows: List[Dict[str, Any]] = []

    # 메인 → 보조 순서
    ordered_types = [
        "pressure",
        "level",
        "flow",
        "current",
        "voltage",
        "inverter",
        "zt",
        "ao",
        "state",
    ]

    for sensor_type in ordered_types:
        limit_n = sensor_type_limit(sensor_type)
        type_rows = usable[usable["sensor_type"] == sensor_type].copy()

        # missing 적고, std가 너무 0에 가깝지 않은 걸 우선
        type_rows = type_rows.sort_values(
            ["missing_rate", "std", "sensor"],
            ascending=[True, False, True]
        )

        selected_rows.extend(type_rows.head(limit_n).to_dict("records"))

    selected_df = pd.DataFrame(selected_rows)
    if len(selected_df) > 0:
        selected_df = selected_df.drop_duplicates(subset=["sensor"])

    # 최소 feature 보장
    if len(selected_df) < min_features:
        already = set(selected_df["sensor"].tolist()) if len(selected_df) > 0 else set()

        fallback_df = usable[~usable["sensor"].isin(already)].copy()
        fallback_df["priority"] = fallback_df["sensor_type"].apply(sensor_type_priority)
        fallback_df = fallback_df.sort_values(
            ["priority", "missing_rate", "std", "sensor"],
            ascending=[True, True, False, True]
        )

        need_n = min_features - len(selected_df)
        fallback_rows = fallback_df.head(need_n)

        if len(selected_df) > 0:
            selected_df = pd.concat([selected_df, fallback_rows], ignore_index=True)
            selected_df = selected_df.drop_duplicates(subset=["sensor"])
        else:
            selected_df = fallback_rows.copy()

    if len(selected_df) > 0:
        selected_df["priority"] = selected_df["sensor_type"].apply(sensor_type_priority)
        selected_df = selected_df.sort_values(
            ["priority", "missing_rate", "std", "sensor"],
            ascending=[True, True, False, True]
        )

    return selected_df.to_dict("records") if len(selected_df) > 0 else []


def main():
    pr = project_root()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_list",
        default=str(pr / "data" / "reports" / "train_station_list.csv"),
        help="CSV of train station list"
    )
    parser.add_argument(
        "--feature_detail",
        default=str(pr / "data" / "reports" / "station_feature_config.csv"),
        help="Detailed feature analysis CSV"
    )
    parser.add_argument(
        "--out_dir",
        default=str(pr / "data" / "reports"),
        help="Output directory"
    )
    parser.add_argument(
        "--min_features",
        type=int,
        default=4,
        help="Minimum selected features per station"
    )
    args = parser.parse_args()

    train_list_path = Path(args.train_list)
    detail_path = Path(args.feature_detail)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_list_path, encoding="utf-8-sig")
    train_stations = set(train_df["station"].astype(str).tolist())

    detail_df = load_feature_detail(detail_path)
    detail_df = detail_df[detail_df["station"].astype(str).isin(train_stations)].copy()

    summary_rows = []
    selected_detail_rows = []

    for station, group in detail_df.groupby("station"):
        selected_rows = select_features_for_station(group, min_features=args.min_features)

        selected_features = [r["sensor"] for r in selected_rows]
        selected_types = [r["sensor_type"] for r in selected_rows]

        summary_rows.append({
            "station": station,
            "candidate_sensor_count": int(len(group)),
            "usable_sensor_count": int((group["usable"] == True).sum()),
            "selected_feature_count": int(len(selected_features)),
            "selected_features": "|".join(selected_features),
            "selected_types": "|".join(selected_types),
        })

        for r in selected_rows:
            r2 = dict(r)
            r2["station"] = station
            r2["selected"] = True
            selected_detail_rows.append(r2)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["selected_feature_count", "station"],
        ascending=[False, True]
    )

    selected_detail_df = pd.DataFrame(selected_detail_rows)
    if len(selected_detail_df) > 0:
        selected_detail_df = selected_detail_df.sort_values(
            ["station", "priority", "missing_rate", "std", "sensor"],
            ascending=[True, True, True, False, True]
        )

    summary_path = out_dir / "station_model_config.csv"
    detail_out_path = out_dir / "station_model_config_detail.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    selected_detail_df.to_csv(detail_out_path, index=False, encoding="utf-8-sig")

    print(f"[OK] model config         -> {summary_path}")
    print(f"[OK] selected detail      -> {detail_out_path}")

    print("\n=== RESULT ===")
    print(f"Stations configured: {len(summary_df)}")
    if len(summary_df) > 0:
        print(summary_df["selected_feature_count"].describe())
        print("\n[Top 10 stations]")
        print(summary_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()