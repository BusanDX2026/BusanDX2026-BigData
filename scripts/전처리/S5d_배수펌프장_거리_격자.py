# -*- coding: utf-8 -*-
"""
S5d. 배수펌프장 접근성 → 100m 격자
- 입력: 00_정합_5186/배수펌프장.gpkg (부산 80개소), 01_격자/grid_100m.gpkg
- 출력: 02_레이어별/배수펌프장_grid.parquet
- 지표: pump_dist_m(최근접 거리), pump_n_500m, pump_n_1000m
- 규약: S0 §6(방향: 거리↑=대응취약↑ → 모델에서 부호 반전), 배수능력/가동기준 데이터 없음 → 거리 근사만
"""
import sys, io, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]   # 저장소 루트 (절대경로 하드코딩 금지)
GG = ROOT / "공공데이터" / "가공데이터"
OUT = GG / "02_레이어별"
REP = GG / "_리포트"
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# S5d. 배수펌프장 접근성 → 100m 격자")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

grid = gpd.read_file(GG / "01_격자" / "grid_100m.gpkg")[["grid_id", "x_cen", "y_cen"]]
pmp = gpd.read_file(GG / "00_정합_5186" / "배수펌프장.gpkg").to_crs(5186)
P(f"격자 {len(grid):,} / 배수펌프장 {len(pmp)}개소")

pxy = np.c_[pmp.geometry.x.values, pmp.geometry.y.values]
gxy = np.c_[grid.x_cen.values, grid.y_cen.values]
tree = cKDTree(pxy)
dist, _ = tree.query(gxy, k=1)
n500 = tree.query_ball_point(gxy, r=500, return_length=True)
n1000 = tree.query_ball_point(gxy, r=1000, return_length=True)

df = pd.DataFrame({
    "grid_id": grid.grid_id.values,
    "pump_dist_m": np.round(dist, 1),
    "pump_n_500m": n500.astype(int),
    "pump_n_1000m": n1000.astype(int),
})
df.to_parquet(OUT / "배수펌프장_grid.parquet", index=False)
P(f"- 최근접 거리(m): {df.pump_dist_m.describe(percentiles=[.5,.9]).round(0).to_dict()}")
P(f"- 1km 내 펌프장 있는 격자: {int((df.pump_n_1000m>0).sum()):,} ({(df.pump_n_1000m>0).mean():.1%})")
P(f"- 500m 내 있는 격자: {int((df.pump_n_500m>0).sum()):,}")
P(f"- 산출: {OUT/'배수펌프장_grid.parquet'}")

(REP / "S5d_배수펌프장.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'S5d_배수펌프장.md'}", flush=True)
