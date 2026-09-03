# -*- coding: utf-8 -*-
"""
S5b. 홍수위험지도(하천범람·도시침수) + 침수흔적도 → 100m 격자
- 입력: 01_격자/grid_100m.gpkg, 00_정합_5186/홍수위험지도_*.gpkg, 00_정합_5186/침수흔적도.gpkg
- 출력: 02_레이어별/홍수위험_grid.parquet, 02_레이어별/침수흔적_grid.parquet
- 규약: S0 §4 (공간미포함=0 / 미관측=NaN), §9 (침수흔적 = 검증 타깃, 피처 파일과 분리)
- 침수심 등급: SEG_CODE N330~N334 = 한강홍수통제소 표준 침수심 구분(N330 최저 ~ N334 최고). 세부 m 구간 추정.
- 속도: dissolve/overlay 대신 sjoin 후보 추출 → 매칭쌍만 intersection.area
"""
import sys, io, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd

ROOT = Path(__file__).resolve().parents[2]   # 저장소 루트 (절대경로 하드코딩 금지)
GG = ROOT / "공공데이터" / "가공데이터"
IN = GG / "00_정합_5186"
OUT = GG / "02_레이어별"
REP = GG / "_리포트"
CRS = 5186
CELL_AREA = 100.0 * 100.0
GRADE = {"N330": 1, "N331": 2, "N332": 3, "N333": 4, "N334": 5}
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# S5b. 홍수위험지도 + 침수흔적도 → 100m 격자")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

grid = gpd.read_file(GG / "01_격자" / "grid_100m.gpkg")[["grid_id", "sgg_cd", "geometry"]].to_crs(CRS)
grid_geom = grid.set_index("grid_id")["geometry"]
P(f"100m 격자: {len(grid):,}")

def poly_to_grid(poly_gdf, grade_col, tag):
    """poly_gdf(등급 폴리곤) → 격자별 {tag}_grade(max), {tag}_area_ratio. sjoin 후보 → 매칭쌍 intersection."""
    pg = poly_gdf[[grade_col, "geometry"]].rename(columns={grade_col: "grade"}).reset_index(drop=True)
    pg["pid"] = np.arange(len(pg))
    j = gpd.sjoin(grid[["grid_id", "geometry"]], pg[["pid", "grade", "geometry"]],
                  how="inner", predicate="intersects").drop(columns="index_right")
    P(f"  {tag}: sjoin 후보쌍 {len(j):,} (격자 {j.grid_id.nunique():,}개)")
    pgi = pg.set_index("pid")["geometry"]
    inter_a = j.apply(lambda r: grid_geom[r.grid_id].intersection(pgi[r.pid]).area, axis=1)
    j = j.assign(a=inter_a.values)
    g = j.groupby("grid_id").agg(**{
        f"{tag}_grade": ("grade", "max"),
        f"{tag}_area_ratio": ("a", lambda s: min(s.sum() / CELL_AREA, 1.0)),
    }).reset_index()
    return g

# ---------- 하천범람 (국가 + 지방, 200년) ----------
ntn = gpd.read_file(IN / "홍수위험지도_국가하천_200.gpkg").to_crs(CRS)
rgn = gpd.read_file(IN / "홍수위험지도_지방하천_200.gpkg").to_crs(CRS)
fluv = gpd.GeoDataFrame(pd.concat([ntn, rgn], ignore_index=True), crs=CRS)
fluv["grade"] = fluv.SEG_CODE.map(GRADE)
gf = poly_to_grid(fluv, "grade", "fluv")
P(f"- 하천범람: 격자 {len(gf):,}개 침수예상범위 교차 (국가+지방 200년)")

# ---------- 도시침수 (100년, 강서구만) ----------
cty = gpd.read_file(IN / "홍수위험지도_도시침수_100.gpkg").to_crs(CRS)
cty["grade"] = cty.SEG_CODE.map(GRADE)
gc = poly_to_grid(cty, "grade", "pluv")
P(f"- 도시침수(내수): 격자 {len(gc):,}개 교차 (강서구 26440 한정 구축)")

hz = grid[["grid_id", "sgg_cd"]].merge(gf, on="grid_id", how="left").merge(gc, on="grid_id", how="left")
hz["fluv_grade"] = hz["fluv_grade"].fillna(0).astype(int)
hz["fluv_area_ratio"] = hz["fluv_area_ratio"].fillna(0.0)
gangseo = hz.sgg_cd.astype(str) == "26440"
hz["pluv_grade"] = np.where(gangseo, hz["pluv_grade"].fillna(0), np.nan)
hz["pluv_area_ratio"] = np.where(gangseo, hz["pluv_area_ratio"].fillna(0.0), np.nan)
hz[["grid_id", "fluv_grade", "fluv_area_ratio", "pluv_grade", "pluv_area_ratio"]].to_parquet(OUT / "홍수위험_grid.parquet", index=False)
P(f"- 하천범람 격자 {int((hz.fluv_grade>0).sum()):,} ({(hz.fluv_grade>0).mean():.1%}), "
  f"등급분포 {hz.loc[hz.fluv_grade>0,'fluv_grade'].value_counts().sort_index().to_dict()}")
P(f"- 도시침수 격자 {int((hz.pluv_grade>0).sum()):,} / 강서 격자 {int(gangseo.sum()):,} (그 외 NaN)")

# ---------- 침수흔적도 (검증 타깃) ----------
P("\n## 침수흔적도 → 격자 (검증 타깃, 피처 아님)")
fs = gpd.read_file(IN / "침수흔적도.gpkg").to_crs(CRS)
for c in ["FLDN_GRD", "FLDN_DOWA", "FLDN_YR"]:
    fs[c] = pd.to_numeric(fs[c], errors="coerce")
fs = fs[["FLDN_GRD", "FLDN_DOWA", "FLDN_YR", "geometry"]].reset_index(drop=True)
fs["tid"] = np.arange(len(fs))
j = gpd.sjoin(grid[["grid_id", "geometry"]], fs[["tid", "FLDN_GRD", "FLDN_DOWA", "FLDN_YR", "geometry"]],
              how="inner", predicate="intersects").drop(columns="index_right")
P(f"  침수흔적 sjoin 후보쌍 {len(j):,} (격자 {j.grid_id.nunique():,}개)")
fsi = fs.set_index("tid")["geometry"]
j = j.assign(a=j.apply(lambda r: grid_geom[r.grid_id].intersection(fsi[r.tid]).area, axis=1))
tr = j.groupby("grid_id").agg(
    trace_area_ratio=("a", lambda s: min(s.sum() / CELL_AREA, 1.0)),
    trace_max_depth=("FLDN_DOWA", "max"),
    trace_max_grade=("FLDN_GRD", "max"),
    trace_last_year=("FLDN_YR", "max"),
    trace_count=("tid", "nunique"),
).reset_index()
out = grid[["grid_id"]].merge(tr, on="grid_id", how="left")
out["trace_flag"] = (out["trace_area_ratio"].fillna(0) > 0).astype(int)
for c in ["trace_area_ratio", "trace_max_depth", "trace_max_grade", "trace_count"]:
    out[c] = out[c].fillna(0)
out.to_parquet(OUT / "침수흔적_grid.parquet", index=False)
P(f"- 침수흔적 격자 {int(out.trace_flag.sum()):,} ({out.trace_flag.mean():.1%})")
P(f"- 침수심 max 분포: {out.loc[out.trace_flag==1,'trace_max_depth'].describe(percentiles=[.5,.9]).round(2).to_dict()}")
P(f"- 최근 침수연도: {out.loc[out.trace_flag==1,'trace_last_year'].value_counts().sort_index().to_dict()}")

m = hz[["grid_id", "fluv_grade"]].merge(out[["grid_id", "trace_flag"]], on="grid_id")
tp = int(((m.fluv_grade > 0) & (m.trace_flag == 1)).sum())
P(f"\n## 참고: 하천범람예상(≥1등급) ∩ 침수흔적 = {tp:,}격자 "
  f"(침수흔적 격자 중 {tp/max(int(out.trace_flag.sum()),1):.1%})")
P("  ※ 정식 검증(AUC·포착률)은 모델링 단계. 여기선 leakage 방지 위해 흔적을 피처와 분리 저장만.")

(REP / "S5b_홍수위험_침수흔적.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'S5b_홍수위험_침수흔적.md'}", flush=True)
