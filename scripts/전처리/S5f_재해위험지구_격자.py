# -*- coding: utf-8 -*-
"""
S5f. 자연재해위험개선지구 → 100m 격자 (비교 baseline, 피처 아님)
- 입력: raw/10_위험_재해위험지구_부산/재해위험지구_부산.csv (71건)
- 출력: 02_레이어별/재해위험지구_grid.parquet  (S6에서 baseline_grid.parquet 로 분리 저장)
        02_레이어별/재해위험지구_지오코딩결과.csv (감사용)
- 방법: 브이월드 검색 API로 '지구명'(+시군구) → 좌표. 실패 시 시군구 중심 근사(정밀도='구').
        지정면적으로 반경 산정(최대 400m) → 반경 내 격자에 플래그.
- 유형코드: 001=침수, 002=붕괴, 003=고립, 006=기타 (행안부 재해위험지구 유형)
- 규약: S0 §5(부산 bbox), §8(키 secrets/), §9(재해지정은 피처 아님 — 우리 지수와 대조용)
"""
import sys, io, glob, math, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, geopandas as gpd
from _vworld import get_key, search_place, polite

ROOT = Path(__file__).resolve().parents[2]   # 저장소 루트 (절대경로 하드코딩 금지)
RAW = ROOT / "공공데이터" / "raw"
GG = ROOT / "공공데이터" / "가공데이터"
OUT = GG / "02_레이어별"
REP = GG / "_리포트"
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# S5f. 재해위험지구 → 100m 격자 (baseline)")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")
KEY = get_key()

f = [x for x in glob.glob(str(RAW / "10_위험_재해위험지구_부산" / "*.csv")) if "부산" in x and "원본" not in x][0]
d = pd.read_csv(f, encoding="utf-8-sig", dtype=str)
d["DSGN_AREA"] = pd.to_numeric(d["DSGN_AREA"], errors="coerce")
d["grade"] = pd.to_numeric(d["DST_RSK_DSTRCT_GRD_CD"], errors="coerce")
d["type_cd"] = d["DST_RSK_DSTRCT_TYPE_CD"].astype(str).str.zfill(3)
d["is_flood"] = d["type_cd"] == "001"
d["active"] = d["RMV_YMD"].isna() | (d["RMV_YMD"].astype(str).str.strip().isin(["", "nan", "None"]))
P(f"재해위험지구: {len(d)}건 | 침수유형 {int(d.is_flood.sum())} | 유효(미해제) {int(d.active.sum())} | 유효+침수 {int((d.is_flood & d.active).sum())}")

# 시군구 중심 (폴백용)
sgg = gpd.read_file(GG / "01_격자" / "부산_시군구.gpkg").to_crs(4326)
sgg_cen = {r.sgg_nm: (r.geometry.centroid.x, r.geometry.centroid.y) for _, r in sgg.iterrows()}

import re
recs = []
for _, r in d.iterrows():
    nm = str(r["DST_RSK_DSTRCT_NM"]).strip()
    gu = str(r["시군구"]).strip()
    base = re.sub(r"\d*지구$|\([^)]*\)", "", nm).strip()
    pt, how = None, "FAIL"
    for label, q in [("지구명", nm), ("구+지구명", f"부산 {gu} {nm}"),
                     ("구+base", f"부산 {gu} {base}" if base else None),
                     ("base", base if base and len(base) >= 2 else None)]:
        if not q:
            continue
        pt = search_place(q, KEY); polite()
        if pt:
            how = "search:" + label; break
    if pt is None and gu in sgg_cen:
        pt = sgg_cen[gu]; how = "구중심근사"
    lon, lat = pt
    recs.append(dict(지구명=nm, 시군구=gu, 유형=r["type_cd"], 등급=r["grade"], 침수=r["is_flood"],
                     유효=r["active"], 지정면적=r["DSGN_AREA"], lon=lon, lat=lat, 정밀도=how))
res = pd.DataFrame(recs)
P(f"- 지오코딩: 지명검색 {(res.정밀도.str.startswith('search')).sum()} / 구중심근사 {(res.정밀도=='구중심근사').sum()} / 실패 {(res.정밀도=='FAIL').sum()}")
res.to_csv(OUT / "재해위험지구_지오코딩결과.csv", index=False, encoding="utf-8-sig")

# ---- 격자 매핑 (반경 버퍼) ----
grid = gpd.read_file(GG / "01_격자" / "grid_100m.gpkg")[["grid_id", "geometry"]].to_crs(5186)
pts = gpd.GeoDataFrame(res.dropna(subset=["lon"]), geometry=gpd.points_from_xy(res.dropna(subset=["lon"]).lon,
                       res.dropna(subset=["lon"]).lat), crs=4326).to_crs(5186)
pts["rad"] = pts["지정면적"].fillna(2500).clip(lower=2500).pow(0.5).div(math.sqrt(math.pi)).clip(upper=400)
pts["rad"] = pts["rad"].where(pts["정밀도"].str.startswith("search"), 150)   # 구중심근사는 반경 축소(대표점만)
buf = pts.copy(); buf["geometry"] = pts.geometry.buffer(pts["rad"])
j = gpd.sjoin(grid, buf, how="inner", predicate="intersects")
agg = j.groupby("grid_id").agg(
    hazdist_flag=("지구명", "size"),
    hazdist_flood=("침수", lambda s: int(any(str(x) == "True" for x in s))),
    hazdist_flood_active=("침수", lambda s: 0),   # 아래서 계산
    hazdist_grade_worst=("등급", "min"),
).reset_index()
fa = j.assign(fa=(j["침수"].astype(str) == "True") & (j["유효"].astype(str) == "True")).groupby("grid_id")["fa"].max().astype(int)
agg["hazdist_flood_active"] = agg["grid_id"].map(fa).fillna(0).astype(int)
names = j.groupby("grid_id")["지구명"].apply(lambda s: "|".join(sorted(set(s)))).rename("hazdist_names")
agg = agg.merge(names, on="grid_id", how="left")

full = gpd.read_file(GG / "01_격자" / "grid_100m.gpkg")[["grid_id", "sgg_nm"]].merge(agg, on="grid_id", how="left")
for c in ["hazdist_flag", "hazdist_flood", "hazdist_flood_active"]:
    full[c] = full[c].fillna(0).astype(int)
full.to_parquet(OUT / "재해위험지구_grid.parquet", index=False)

P(f"\n## 검증")
P(f"- 재해위험지구 격자 {int((full.hazdist_flag>0).sum())} | 침수유형 격자 {int((full.hazdist_flood>0).sum())} | 유효+침수 격자 {int((full.hazdist_flood_active>0).sum())}")
P(f"- 자치구별 재해위험지구(원본): {d.groupby('시군구').size().to_dict()}")
P(f"- 산출: {OUT/'재해위험지구_grid.parquet'} (S6 → baseline_grid.parquet), {OUT/'재해위험지구_지오코딩결과.csv'}")
P("  ※ 정밀 위치 불확실분은 '구중심근사'로 표기. 우리 지수 vs 행정지정 대조는 자치구/행정동 단위 우선.")

(REP / "S5f_재해위험지구.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'S5f_재해위험지구.md'}", flush=True)
