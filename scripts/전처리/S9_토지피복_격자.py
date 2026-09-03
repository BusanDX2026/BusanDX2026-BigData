# -*- coding: utf-8 -*-
"""
S9. 토지피복지도 세분류 → 100m 격자 (불투수면적률 등)

방법
  310,605개 폴리곤을 81,435개 격자에 overlay 하면 매우 느리므로,
  **10m 래스터로 rasterize 후 10×10 블록 평균**으로 면적비를 구한다(정확도 동일, 수십 배 빠름).
  래스터 격자는 S5c/M3 와 동일 정렬(원점 = 격자 좌하단 100m 스냅).

산출 지표
  imperv_ratio  시가화건조지역(l1=100) 비율 ← 내수침수 유출량의 직접 대리변수
  agri_ratio    농업지역(200) — '저지대이지만 농경지'(강서 삼각주) 식별용, 모델 이슈 #5 대응
  paddy_ratio   논(l2=210) — 일시 저류 기능이 있어 침수 완충
  forest_ratio  산림(300) / grass_ratio 초지(400) / bare_ratio 나지(600) / water_ratio 수역(700)
  road_ratio    교통지역(l2=150) — 도로는 우수 유출 경로

- 입력: raw/13_토지피복_세분류_부산/토지피복_세분류_부산.gpkg, 01_격자/grid_100m.gpkg
- 출력: 02_레이어별/토지피복_grid.parquet
"""
import sys, io, time, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"
SRC = ROOT / "공공데이터" / "raw" / "13_토지피복_세분류_부산" / "토지피복_세분류_부산.gpkg"
OUT = GG / "02_레이어별"
REP = GG / "_리포트"
SUB = 10                      # 10m 서브픽셀 → 100m 격자당 10×10
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# S9. 토지피복 → 100m 격자")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

grid = gpd.read_file(GG / "01_격자" / "grid_100m.gpkg")[["grid_id", "x_cen", "y_cen", "sgg_nm"]]
lc = gpd.read_file(SRC).to_crs(5186)
P(f"격자 {len(grid):,} · 토지피복 폴리곤 {len(lc):,}")
lc["l1"] = pd.to_numeric(lc.l1_code, errors="coerce").fillna(0).astype(int)
lc["l2"] = pd.to_numeric(lc.l2_code, errors="coerce").fillna(0).astype(int)

# --- 래스터 정의 (격자와 동일 정렬) ---
CELL = 100
x_min = grid.x_cen.min() - CELL / 2
y_max = grid.y_cen.max() + CELL / 2
ncol = int(round((grid.x_cen.max() + CELL / 2 - x_min) / CELL))
nrow = int(round((y_max - (grid.y_cen.min() - CELL / 2)) / CELL))
RW, RH = ncol * SUB, nrow * SUB
tr = from_origin(x_min, y_max, CELL / SUB, CELL / SUB)
P(f"래스터 {RH}x{RW} @ {CELL/SUB:.0f}m (격자 {nrow}x{ncol})")

def frac(mask_gdf, label):
    """폴리곤 집합 → 10m 래스터 → 100m 격자 면적비"""
    if len(mask_gdf) == 0:
        return np.zeros(len(grid), dtype="float32")
    t0 = time.perf_counter()
    r = rasterize(((g, 1) for g in mask_gdf.geometry),out_shape=(RH, RW), transform=tr,
                  fill=0, dtype="uint8", all_touched=False)
    blocks = r.reshape(nrow, SUB, ncol, SUB).mean(axis=(1, 3)).astype("float32")
    P(f"  {label:<14} 폴리곤 {len(mask_gdf):>7,} → {time.perf_counter()-t0:5.1f}s")
    return blocks

DEFS = [
    ("imperv_ratio", lc.l1 == 100, "시가화건조지역"),
    ("agri_ratio",   lc.l1 == 200, "농업지역"),
    ("paddy_ratio",  lc.l2 == 210, "논"),
    ("forest_ratio", lc.l1 == 300, "산림지역"),
    ("grass_ratio",  lc.l1 == 400, "초지"),
    ("bare_ratio",   lc.l1 == 600, "나지"),
    ("water_ratio",  lc.l1 == 700, "수역"),
    ("road_ratio",   lc.l2 == 150, "교통지역"),
]
col = ((grid.x_cen.values - x_min) / CELL).astype(int)
row = ((y_max - grid.y_cen.values) / CELL).astype(int)
assert col.min() >= 0 and col.max() < ncol and row.min() >= 0 and row.max() < nrow

out = pd.DataFrame({"grid_id": grid.grid_id.values})
P("\n래스터화 진행:")
for name, sel, kor in DEFS:
    blocks = frac(lc[sel], kor)
    out[name] = blocks[row, col]

# 토지피복 자체가 없는 격자(해상 등) 표시
cov = sum(out[n] for n, _, _ in DEFS)
out["lc_covered"] = (cov > 0.5).astype(int)
out.to_parquet(OUT / "토지피복_grid.parquet", index=False)

P(f"\n## 검증")
P(f"- 피복 합계 중앙값 {cov.median():.3f} (1.0에 가까워야 정상, 해상 격자는 0)")
P(f"- 토지피복 있는 격자 {int(out.lc_covered.sum()):,} / {len(out):,} ({out.lc_covered.mean():.1%})")
P(f"- imperv_ratio: 평균 {out.imperv_ratio.mean():.3f}, >0.5 격자 {int((out.imperv_ratio>0.5).sum()):,}")
g2 = grid.assign(imperv=out.imperv_ratio.values, agri=out.agri_ratio.values)
P("\n자치구별 평균 불투수율 / 농업지역 비율:")
t = g2.groupby("sgg_nm")[["imperv", "agri"]].mean().sort_values("imperv", ascending=False)
for nm, r in t.iterrows():
    P(f"  {nm:<7} 불투수 {r.imperv:.3f} | 농업 {r.agri:.3f}")
P("\n→ 강서구가 '농업 비율 최고 + 불투수 낮음'으로 나오면 이슈 #5(저지대=안전 역방향 학습) 해소 근거가 생긴다.")
P(f"- 산출: {(OUT/'토지피복_grid.parquet').relative_to(ROOT)}")

(REP / "S9_토지피복.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'S9_토지피복.md'}", flush=True)
