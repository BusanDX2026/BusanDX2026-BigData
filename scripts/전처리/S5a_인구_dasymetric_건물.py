# -*- coding: utf-8 -*-
"""
S5a. 인구 dasymetric 배분(연면적 가중) + 건물 지표 → 100m 격자
- 입력:
    01_격자/grid_100m.gpkg
    00_정합_5186/sgis_격자1km.gpkg
    raw/02_인구_SGIS격자/**/*인구*1K.csv        (to_in_001 총인구, in_age_014~019 고령)
    raw/03_건물_GIS건물통합정보_부산/**/AL_D010_26_20260809.shp  (A9 주용도, A14 연면적, A26/A27 층수)
- 출력: 02_레이어별/인구건물_grid.parquet  (grid_id 키)
- 방법: 1km 총인구를 셀 내 건물 '주거 연면적' 비율로 배분(주거 없으면 전체 연면적), 건물 대표점→100m 격자 합산
- 규약: S0 §4(공간미포함=0), §9(총량 보존 ±0.1%)
"""
import sys, io, glob, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd

ROOT = Path(__file__).resolve().parents[2]   # 저장소 루트 (절대경로 하드코딩 금지)
RAW = ROOT / "공공데이터" / "raw"
GG = ROOT / "공공데이터" / "가공데이터"
OUT = GG / "02_레이어별"; OUT.mkdir(parents=True, exist_ok=True)
REP = GG / "_리포트"
CRS = 5186
log = []
def P(s=""):
    print(s); log.append(str(s))

P("# S5a. 인구 dasymetric(연면적 가중) + 건물 지표")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

grid = gpd.read_file(GG / "01_격자" / "grid_100m.gpkg")[["grid_id", "geometry"]].to_crs(CRS)
P(f"100m 격자: {len(grid):,}셀")

# ---------- 1km 인구 ----------
g1k = gpd.read_file(GG / "00_정합_5186" / "sgis_격자1km.gpkg").to_crs(CRS)   # GRID_CD, geometry
csvs = sorted(glob.glob(str(RAW / "02_인구_SGIS격자" / "**" / "*인구*1K.csv"), recursive=True))
pop = pd.concat([pd.read_csv(c, encoding="cp949", dtype={"격자코드": str, "통계항목": str})
                 for c in csvs], ignore_index=True)
pop["통계값"] = pd.to_numeric(pop["통계값"], errors="coerce").fillna(0)
piv = pop.pivot_table(index="격자코드", columns="통계항목", values="통계값", aggfunc="sum").fillna(0)
old_cols = [f"in_age_{i:03d}" for i in range(14, 20) if f"in_age_{i:03d}" in piv.columns]   # 65~94세
age_cols = [c for c in piv.columns if c.startswith("in_age_0") and c[-3:].isdigit() and int(c[-3:]) <= 19]
piv["pop_total"] = piv.get("to_in_001", 0)
# ⚠ SGIS는 소수 카운트를 5단위 반올림/보호처리 → '연령별 합 ≠ 총인구'(1km 격자 6%에서 65+ > 총인구).
#   따라서 65+ 인구를 절대값으로 쓰지 않고 '연령 내부 비율'로 환산해 총인구에 곱한다 (반올림 편향 상쇄, 항상 65+ ≤ 총인구 보장).
_o65 = piv[old_cols].sum(axis=1) if old_cols else 0
_aall = piv[age_cols].sum(axis=1) if age_cols else 0
piv["old_share"] = np.where(np.asarray(_aall) > 0, np.asarray(_o65) / np.maximum(np.asarray(_aall), 1e-9), 0.0)
piv["old_share"] = np.clip(piv["old_share"], 0.0, 1.0)
piv["pop_65plus"] = piv["pop_total"] * piv["old_share"]
n_incons = int((np.asarray(_o65) > piv["pop_total"].values).sum())
g1k = g1k.merge(piv[["pop_total", "pop_65plus"]], left_on="GRID_CD", right_index=True, how="left").fillna({"pop_total": 0, "pop_65plus": 0})
P(f"1km 격자: {len(g1k):,}셀, 총인구 합 {g1k.pop_total.sum():,.0f}, 65+ 합 {g1k.pop_65plus.sum():,.0f}")
P(f"- ⚠ SGIS 원본 불일치(연령합 65+ > 총인구) 1km 격자 {n_incons:,}건 → 연령내부비율 방식으로 보정 (65+ ≤ 총인구 보장)")

# ---------- 건물 ----------
bshp = glob.glob(str(RAW / "03_건물_GIS건물통합정보_부산" / "**" / "AL_D010_26_20260809.shp"), recursive=True)[0]
b = gpd.read_file(bshp, columns=["A9", "A12", "A14", "A26", "A27"], encoding="cp949")
b = b.to_crs(CRS)
b["A12"] = pd.to_numeric(b["A12"], errors="coerce").fillna(0.0)       # 건축면적 m²
b["A14"] = pd.to_numeric(b["A14"], errors="coerce").fillna(0.0)       # 연면적 m²
b["A26"] = pd.to_numeric(b["A26"], errors="coerce").fillna(0)          # 지상층수
b["A27"] = pd.to_numeric(b["A27"], errors="coerce").fillna(0)          # 지하층수
n_all = len(b)
b = b[(b.A14 > 0) & b.geometry.notna() & ~b.geometry.is_empty].copy()

# 연면적 이상치 정제 (S0 §5): 건물 외피(max(건축면적,footprint) × 총층수) 의 5배 초과 → 손상값
b["_foot"] = b.geometry.area
env = np.maximum(b["A12"], b["_foot"]) * (b["A26"] + b["A27"].clip(lower=0) + 1)
bad = (b["A14"] > env * 5) & (env > 0)
b["A14_raw"] = b["A14"]
b.loc[bad, "A14"] = np.minimum(env[bad] * 1.2, 400000)
b["A14"] = b["A14"].clip(upper=400000)   # 하드 상한 (엘시티 랜드마크타워 연면적 ≈ 34만 m²)
P(f"\n건물: 전체 {n_all:,} → 유효(연면적>0) {len(b):,}")
P(f"- 연면적 이상치 정제: 외피 5배 초과 {int(bad.sum())}건 + 40만m² 초과 clip. "
  f"정제 전/후 연면적 합 {b['A14_raw'].sum()/1e6:.1f} / {b['A14'].sum()/1e6:.1f} 백만 m²")
RESID_KW = ("주택", "아파트", "공동주택", "다세대", "연립", "다가구", "기숙사")
b["is_resid"] = b["A9"].fillna("").astype(str).apply(lambda s: any(k in s for k in RESID_KW))
b["resid_area"] = np.where(b.is_resid, b.A14, 0.0)
b["rep"] = b.geometry.representative_point()
P(f"- 주거 건물 {b.is_resid.sum():,} ({b.is_resid.mean():.1%}), 주거 연면적 합 {b.resid_area.sum()/1e6:.2f} km²·층")

# 건물 대표점에 1km GRID_CD, 100m grid_id 부여
bpts = gpd.GeoDataFrame(b[["A14", "A26", "A27", "is_resid", "resid_area"]], geometry=b["rep"].values, crs=CRS)
bpts = bpts.sjoin(g1k[["GRID_CD", "pop_total", "pop_65plus", "geometry"]], how="left", predicate="within").drop(columns="index_right")
bpts = bpts.sjoin(grid, how="left", predicate="within").drop(columns="index_right")
n_no1k = bpts.GRID_CD.isna().sum(); n_no100 = bpts.grid_id.isna().sum()
P(f"- 건물 대표점 1km 미매칭 {n_no1k:,} / 100m 미매칭 {n_no100:,} (부산 경계 밖 건물)")
bpts = bpts.dropna(subset=["grid_id"])

# ---------- dasymetric 배분 ----------
# 각 1km 셀: 배분 가중 = resid_area (셀 내 주거 연면적>0 이면), else A14
cell_resid = bpts.groupby("GRID_CD")["resid_area"].transform("sum")
bpts["w"] = np.where(cell_resid > 0, bpts["resid_area"], bpts["A14"])
cell_w = bpts.groupby("GRID_CD")["w"].transform("sum")

# 건물 커버리지 불량 1km 셀 감지: 1인당 건물 연면적 < 10 m² → 건물 레이어가 인구를 못 담음
percap = cell_w / bpts["pop_total"].replace(0, np.nan)
undercov = set(bpts.loc[(percap < 10) & (bpts["pop_total"] > 0), "GRID_CD"].dropna().unique())
good = ~bpts["GRID_CD"].isin(undercov)
bpts["frac"] = 0.0
gw = bpts.loc[good].groupby("GRID_CD")["w"].transform("sum")
bpts.loc[good, "frac"] = np.where(gw > 0, bpts.loc[good, "w"] / gw, 0.0)
bpts["alloc_pop"] = bpts["frac"] * bpts["pop_total"].fillna(0)
bpts["alloc_65"] = bpts["frac"] * bpts["pop_65plus"].fillna(0)
P(f"- 건물 dasymetric 적용 1km 셀 {bpts.loc[good,'GRID_CD'].nunique():,}, "
  f"커버리지 불량(→균등배분) {len(undercov):,}")

# 건물 없는/불량 1km 셀의 인구 → 해당 1km와 겹치는 100m 육지셀에 균등 배분
served = set(bpts.loc[good, "GRID_CD"].dropna().unique())
miss = g1k[(~g1k.GRID_CD.isin(served)) & (g1k.pop_total > 0)]
extra_rows = []
if len(miss):
    j = gpd.sjoin(grid, miss[["GRID_CD", "pop_total", "pop_65plus", "geometry"]], how="inner", predicate="intersects")
    cnt = j.groupby("GRID_CD").size()
    for gid, r in j.iterrows():
        k = cnt[r.GRID_CD]
        extra_rows.append((r.grid_id, r.pop_total / k, r.pop_65plus / k))
    P(f"- 건물無 인구有 1km 셀 {len(miss)}개 → 100m 육지셀 균등 배분 ({len(extra_rows)}건)")

# ---------- 100m 격자 집계 ----------
agg = bpts.groupby("grid_id").agg(
    pop=("alloc_pop", "sum"),
    pop_65=("alloc_65", "sum"),
    bldg_cnt=("A14", "size"),
    floor_area=("A14", "sum"),
    resid_floor_area=("resid_area", "sum"),
    basement_bldg=("A27", lambda s: int((s > 0).sum())),
    max_floors=("A26", "max"),
).reset_index()
if extra_rows:
    ex = pd.DataFrame(extra_rows, columns=["grid_id", "pop_x", "pop65_x"]).groupby("grid_id").sum().reset_index()
    agg = agg.merge(ex, on="grid_id", how="outer")
    agg["pop"] = agg["pop"].fillna(0) + agg["pop_x"].fillna(0)
    agg["pop_65"] = agg["pop_65"].fillna(0) + agg["pop65_x"].fillna(0)
    agg = agg.drop(columns=["pop_x", "pop65_x"])

out = grid[["grid_id"]].merge(agg, on="grid_id", how="left")
num_cols = [c for c in out.columns if c != "grid_id"]
out[num_cols] = out[num_cols].fillna(0)
# pop_density(=pop/0.01)는 pop과 완전중복(r=1.0)이라 생성하지 않음 — 격자면적이 모두 동일하기 때문.
out["pop_65"] = np.minimum(out["pop_65"], out["pop"])          # 안전장치
out["old_ratio"] = np.where(out["pop"] > 0, out["pop_65"] / out["pop"], 0.0)
assert out["old_ratio"].max() <= 1.0 + 1e-9, "old_ratio > 1 — 고령 배분 로직 점검 필요"
assert (out["pop_65"] <= out["pop"] + 1e-6).all(), "pop_65 > pop"
out.to_parquet(OUT / "인구건물_grid.parquet", index=False)

# ---------- 검증 ----------
src_total = g1k.pop_total.sum()
src_in_busan = g1k.loc[g1k.GRID_CD.isin(served) | g1k.GRID_CD.isin(miss.GRID_CD), "pop_total"].sum()
alloc_total = out["pop"].sum()
P(f"\n## 검증 (S0 §9 총량 보존)")
P(f"- 1km 총인구 합계        : {src_total:,.0f}")
P(f"- 배분된 100m 인구 합계   : {alloc_total:,.0f}")
P(f"- 차이                   : {alloc_total - src_total:,.0f} ({(alloc_total/src_total-1)*100:+.3f}%)")
P(f"- 미배분(건물없음+경계밖) : {src_total - alloc_total:,.0f}  (부산 경계 밖 1km 셀 인구 = 정상 손실)")
P(f"- 부산 2024 주민등록인구 약 328만 대비 SGIS 총인구 합 {src_total/1e4:.1f}만")
P(f"- 인구 있는 격자 {int((out['pop']>0).sum()):,} / 건물 있는 격자 {int((out['bldg_cnt']>0).sum()):,} / 전체 {len(out):,}")
P(f"- 격자 최대 인구 {out['pop'].max():,.0f}명 (1ha). 최상위 셀 1인당 주거연면적 검증:")
_v = out[(out['pop'] > 30) & (out.resid_floor_area > 0)]
_pc = (_v.resid_floor_area / _v['pop'])
P(f"    1인당 주거연면적 5%/중앙/95% = {_pc.quantile(.05):.1f} / {_pc.median():.1f} / {_pc.quantile(.95):.1f} m² (한국 평균 ~30-35 → 배분 타당)")
P(f"- old_ratio 범위 {out.old_ratio.min():.3f}~{out.old_ratio.max():.3f} (≤1 보장)")
P(f"- 산출: {OUT/'인구건물_grid.parquet'}  컬럼 {list(out.columns)}")

(REP / "S5a_인구건물.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'S5a_인구건물.md'}")
