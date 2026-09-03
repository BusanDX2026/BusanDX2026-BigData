# -*- coding: utf-8 -*-
"""
S4. 100m fishnet 격자 생성 (EPSG:5186)
- 입력: 공공데이터/가공데이터/01_격자/부산_경계.gpkg, 부산_시군구.gpkg, 부산_행정동.gpkg
- 출력: 공공데이터/가공데이터/01_격자/grid_100m.gpkg
- 규약: S0 §2 (100m, 100의 배수 스냅, 부산 intersects, grid_id=f"{x//100}_{y//100}", x_cen/y_cen/sgg_cd/adm_cd 보존)
"""
import sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from shapely import STRtree

ROOT = Path(__file__).resolve().parents[2]   # 저장소 루트 (절대경로 하드코딩 금지)
GG = ROOT / "공공데이터" / "가공데이터"
GRID = GG / "01_격자"
REP = GG / "_리포트"
CRS = 5186
CELL = 100
log = []
def P(s=""):
    print(s); log.append(str(s))

P("# S4. 100m fishnet 격자 생성")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}")

bnd = gpd.read_file(GRID / "부산_경계.gpkg").to_crs(CRS)
poly = bnd.geometry.iloc[0]
minx, miny, maxx, maxy = poly.bounds
x0 = np.floor(minx / CELL) * CELL
y0 = np.floor(miny / CELL) * CELL
x1 = np.ceil(maxx / CELL) * CELL
y1 = np.ceil(maxy / CELL) * CELL
nx = int((x1 - x0) / CELL)
ny = int((y1 - y0) / CELL)
P(f"\n## 격자 범위 (100m 스냅)")
P(f"- bbox: X {x0:.0f}~{x1:.0f}, Y {y0:.0f}~{y1:.0f}  → {nx} x {ny} = {nx*ny:,} 후보셀")

# 후보 셀 생성 + 경계 intersects 필터 (STRtree로 가속)
cells, ids, xc, yc = [], [], [], []
for ix in range(nx):
    xmin = x0 + ix * CELL
    for iy in range(ny):
        ymin = y0 + iy * CELL
        cells.append(box(xmin, ymin, xmin + CELL, ymin + CELL))
        ids.append(f"{int(xmin//CELL)}_{int(ymin//CELL)}")
        xc.append(xmin + CELL / 2)
        yc.append(ymin + CELL / 2)
P(f"- 후보 {len(cells):,}개 생성, 경계 교차 판정 중...")

tree = STRtree(cells)
hit = tree.query(poly, predicate="intersects")
hit = np.sort(hit)
P(f"- 부산 경계 intersects: {len(hit):,}셀")

g = gpd.GeoDataFrame(
    {"grid_id": [ids[i] for i in hit],
     "x_cen": [xc[i] for i in hit],
     "y_cen": [yc[i] for i in hit]},
    geometry=[cells[i] for i in hit], crs=CRS)
assert g.grid_id.is_unique, "grid_id 중복!"

# 중심점으로 sgg_cd / adm_cd 부여
cen = gpd.GeoDataFrame(g[["grid_id"]], geometry=gpd.points_from_xy(g.x_cen, g.y_cen), crs=CRS)
sgg = gpd.read_file(GRID / "부산_시군구.gpkg").to_crs(CRS)[["sgg_cd", "sgg_nm", "geometry"]]
dong = gpd.read_file(GRID / "부산_행정동.gpkg").to_crs(CRS)[["ADM_CD", "ADM_NM", "geometry"]]
cen = cen.sjoin(sgg, how="left", predicate="within").drop(columns="index_right")
cen = cen.sjoin(dong, how="left", predicate="within").drop(columns="index_right")
cen = cen.drop_duplicates("grid_id")
g = g.merge(cen.drop(columns="geometry"), on="grid_id", how="left")
g = g.rename(columns={"ADM_CD": "adm_cd", "ADM_NM": "adm_nm"})

n_no_sgg0 = g.sgg_cd.isna().sum()
# 중심점이 육지 밖인 해안 partial 셀 → 최대 교차면적 구/동으로 폴백
miss = g[g.sgg_cd.isna()]
if len(miss):
    ov = gpd.overlay(g[["grid_id", "geometry"]].iloc[miss.index], sgg, how="intersection")
    ov["a"] = ov.area
    best = ov.sort_values("a").drop_duplicates("grid_id", keep="last").set_index("grid_id")
    g.loc[g.sgg_cd.isna(), "sgg_cd"] = g.loc[g.sgg_cd.isna(), "grid_id"].map(best["sgg_cd"])
    g.loc[g.sgg_nm.isna(), "sgg_nm"] = g.loc[g.sgg_nm.isna(), "grid_id"].map(best["sgg_nm"])
    ovd = gpd.overlay(g[["grid_id", "geometry"]].iloc[miss.index], dong, how="intersection")
    ovd["a"] = ovd.area
    bd = ovd.sort_values("a").drop_duplicates("grid_id", keep="last").set_index("grid_id")
    g.loc[g.adm_cd.isna(), "adm_cd"] = g.loc[g.adm_cd.isna(), "grid_id"].map(bd["ADM_CD"])
    g.loc[g.adm_nm.isna(), "adm_nm"] = g.loc[g.adm_nm.isna(), "grid_id"].map(bd["ADM_NM"])

n_no_sgg = g.sgg_cd.isna().sum()
n_no_adm = g.adm_cd.isna().sum()
P(f"\n## 행정구역 부여 (격자 중심점 within, 결측은 최대교차면적 폴백)")
P(f"- 중심점 within 실패 {n_no_sgg0} → 폴백 후 sgg_cd 결측 {n_no_sgg} / adm_cd 결측 {n_no_adm}")
P("- 구별 셀 수:")
for r in g.groupby("sgg_nm").size().sort_values(ascending=False).items():
    P(f"    {r[0]}: {r[1]:,}")

g = g[["grid_id", "x_cen", "y_cen", "sgg_cd", "sgg_nm", "adm_cd", "adm_nm", "geometry"]]
g.to_file(GRID / "grid_100m.gpkg", driver="GPKG")

P(f"\n## 검증 (S0 §2·§9)")
P(f"- 총 셀: {len(g):,}  (부산 면적 785.6km² / 0.01km² ≈ 78,560 + 경계 partial)")
P(f"- grid_id 유일성: {g.grid_id.is_unique}")
P(f"- CRS: EPSG:{g.crs.to_epsg()}")
P(f"- 원점 스냅: x0={x0:.0f}, y0={y0:.0f} (both %100=={x0%100==0 and y0%100==0})")
P(f"- 산출: {GRID/'grid_100m.gpkg'}")

(REP / "S4_격자.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'S4_격자.md'}")
