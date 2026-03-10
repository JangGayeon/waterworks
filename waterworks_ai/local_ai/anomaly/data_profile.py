import pandas as pd
from collections import defaultdict

FILE = r"C:\waterworks\waterworks_ai\data\raw\sensor_csv\20230101.csv"

df = pd.read_csv(FILE, encoding='cp949')

time_col = df.columns[0]

station_cols = df.columns[1:]

station_map = defaultdict(list)

for col in station_cols:
    station = col.split(".")[0]
    station_map[station].append(col)

station_rows = []
sensor_rows = []

for station, cols in station_map.items():

    sub = df[cols]

    missing_rate = sub.isna().mean().mean()

    station_rows.append({
        "station": station,
        "sensor_count": len(cols),
        "missing_rate": round(missing_rate,4)
    })

    for c in cols:
        sensor_rows.append({
            "station": station,
            "sensor": c.split(".")[-1],
            "missing_rate": round(df[c].isna().mean(),4)
        })

station_df = pd.DataFrame(station_rows)
sensor_df = pd.DataFrame(sensor_rows)

station_df.to_csv("station_summary.csv", index=False)
sensor_df.to_csv("sensor_summary.csv", index=False)

print("Stations:", len(station_df))
print(station_df.sort_values("missing_rate"))