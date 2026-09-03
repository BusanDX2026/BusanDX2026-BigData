# -*- coding: utf-8 -*-
"""
S5c. DEM 파생지표 → 100m 격자
- 입력: 00_정합_5186/DEM_30m_부산_5186.tif, 01_격자/grid_100m.gpkg
- 출력: 02_레이어별/지형_grid.parquet  (grid_id 키)
- 지표: elev_mean, elev_min, slope_mean(deg), lowland5_ratio(≤5m), lowland3_ratio(≤3m), tpi(주변 5×5=500m 대비 상대표고)
- 방법: 30m DEM → 경사·저지대 이진 계산 → 100m 격자 좌표계로 average 리샘플 (grid_id 규칙과 동일 정렬)
- 규약: S0 §4(no-data 마스킹), §5(음수 표고 유지)
"""
import sys, io, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[2]   # 저장소 루트 (절대경로 하드코딩 금지)
GG = ROOT / "공공데이터" / "가공데이터"
IN = GG / "00_정합_5186"
OUT = GG / "02_레이어별"
REP = GG / "_리포트"
CELL = 100
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# S5c. DEM 파생지표 → 100m 격자")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

grid = gpd.read_file(GG / "01_격자" / "grid_100m.gpkg")[["grid_id", "x_cen", "y_cen"]]
P(f"100m 격자: {len(grid):,}")

with rasterio.open(IN / "DEM_30m_부산_5186.tif") as ds:
    dem = ds.read(1).astype("float32")
    nd = ds.nodata
    src_t, src_crs = ds.transform, ds.crs
    px = src_t.a
dem[dem == nd] = np.nan
P(f"DEM 30m: shape {dem.shape}, 픽셀 {px:.1f}m, 유효 {np.isfinite(dem).mean():.1%}")

# 경사 (도) — 중심차분
gy, gx = np.gradient(dem, px, px)
slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2))).astype("float32")
low5 = np.where(np.isfinite(dem), (dem <= 5).astype("float32"), np.nan)
low3 = np.where(np.isfinite(dem), (dem <= 3).astype("float32"), np.nan)

# ---- 100m 목표 그리드 정의 (grid_id 규칙과 동일: x//100, y//100) ----
xs = np.sort(grid.x_cen.unique()); ys = np.sort(grid.y_cen.unique())
x_min = grid.x_cen.min() - CELL / 2
y_max = grid.y_cen.max() + CELL / 2
ncol = int(round((grid.x_cen.max() + CELL / 2 - x_min) / CELL))
nrow = int(round((y_max - (grid.y_cen.min() - CELL / 2)) / CELL))
dst_t = from_origin(x_min, y_max, CELL, CELL)
P(f"목표 100m 래스터: {nrow} x {ncol}, 원점 ({x_min:.0f}, {y_max:.0f})")

def resample(arr, how):
    out = np.full((nrow, ncol), np.nan, dtype="float32")
    reproject(arr, out, src_transform=src_t, src_crs=src_crs,
              dst_transform=dst_t, dst_crs=src_crs,
              src_nodata=np.nan, dst_nodata=np.nan, resampling=how)
    return out

r_mean = resample(dem, Resampling.average)
r_min = resample(dem, Resampling.min)
r_slope = resample(slope, Resampling.average)
r_low5 = resample(low5, Resampling.average)
r_low3 = resample(low3, Resampling.average)

# TPI(지형위치지수) = 격자 평균표고 − 주변 5×5(=500m) 창 평균표고. 음수 = 주변보다 낮은 국지 저지대.
# ※ 부산은 절대표고 저지대(강서 삼각주)가 농경지·미개발이라 침수흔적과 무관 →
#   '상대'표고인 TPI가 침수 설명력이 훨씬 높음 (전처리 코드리뷰 검증결과 참조).
from scipy.ndimage import uniform_filter
WIN = 5
neigh = uniform_filter(np.where(np.isfinite(r_mean), r_mean, 0), size=WIN)
cnt = uniform_filter(np.isfinite(r_mean).astype("float32"), size=WIN)
neigh = np.where(cnt > 0, neigh / np.maximum(cnt, 1e-6), np.nan)
r_tpi = (r_mean - neigh).astype("float32")

# ---- 격자 좌표 → 래스터 인덱스 (clip으로 조용히 덮지 않고 assert로 정렬 검증) ----
col = ((grid.x_cen.values - x_min) / CELL).astype(int)
row = ((y_max - grid.y_cen.values) / CELL).astype(int)
assert col.min() >= 0 and col.max() < ncol, f"열 인덱스 이탈: {col.min()}~{col.max()} (ncol={ncol})"
assert row.min() >= 0 and row.max() < nrow, f"행 인덱스 이탈: {row.min()}~{row.max()} (nrow={nrow})"

def pick(r):
    return r[row, col]

df = pd.DataFrame({
    "grid_id": grid.grid_id.values,
    "elev_mean": pick(r_mean),
    "elev_min": pick(r_min),
    "slope_mean": pick(r_slope),
    "lowland5_ratio": pick(r_low5),
    "lowland3_ratio": pick(r_low3),
    "tpi": pick(r_tpi),
})
n_nan = df.elev_mean.isna().sum()
P(f"\n- DEM 값 없는 격자(해상·경계밖): {n_nan:,} ({n_nan/len(df):.1%}) → NaN 유지 (S0 §4)")
df.to_parquet(OUT / "지형_grid.parquet", index=False)

v = df.dropna(subset=["elev_mean"])
P(f"- elev_mean: {v.elev_mean.describe(percentiles=[.1,.5,.9]).round(1).to_dict()}")
P(f"- slope_mean(deg): {v.slope_mean.describe(percentiles=[.5,.9]).round(1).to_dict()}")
P(f"- 저지대(≤5m) 비율>0.5 격자: {int((v.lowland5_ratio>0.5).sum()):,}  / TPI<-1(주변보다 1m+ 낮음): {int((v.tpi<-1).sum()):,}")
P(f"- 산출: {OUT/'지형_grid.parquet'}  컬럼 {list(df.columns)}")

(REP / "S5c_지형.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'S5c_지형.md'}", flush=True)
