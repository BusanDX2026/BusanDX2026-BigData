# -*- coding: utf-8 -*-
"""
S2. 좌표계 통일(EPSG:5186) + 부산 경계/행정동 생성 + 벡터 레이어 부산 클립
- 입력: 공공데이터/raw/ , 공공데이터/가공데이터/_해제/ (S1 산출 홍수위험지도)
- 출력: 공공데이터/가공데이터/00_정합_5186/*.gpkg , 01_격자/부산_경계.gpkg·부산_행정동.gpkg·부산_시군구.gpkg
- 규약: 문서/S0_작업규약.md  (§1 CRS 5186, §3 부산 경계 SGIS 21 필터, §9 총량 보존)
- 래스터(DEM)는 S3에서 별도 처리
"""
import sys, io, glob, warnings, csv
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[2]   # 저장소 루트 (절대경로 하드코딩 금지)
RAW = ROOT / "공공데이터" / "raw"
GG = ROOT / "공공데이터" / "가공데이터"
HAE = GG / "_해제"
OUT = GG / "00_정합_5186"
GRID = GG / "01_격자"
REP = GG / "_리포트"
for d in (OUT, GRID, REP):
    d.mkdir(parents=True, exist_ok=True)

CRS = 5186
log = []
def P(s=""):
    print(s); log.append(str(s))

def g1(pat):
    r = glob.glob(str(RAW / pat), recursive=True) or glob.glob(str(GG / pat), recursive=True)
    if not r:
        raise FileNotFoundError(pat)
    return r[0]

# 부산 자치구 이름 → 행정표준 시군구코드(26xxx)
SGG = {"중구":"26110","서구":"26140","동구":"26170","영도구":"26200","부산진구":"26230",
       "동래구":"26260","남구":"26290","북구":"26320","해운대구":"26350","사하구":"26380",
       "금정구":"26410","강서구":"26440","연제구":"26470","수영구":"26500","사상구":"26530","기장군":"26710"}

P("# S2. 좌표계 통일(EPSG:5186) + 부산 클립")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}")
P()

# ---------- 1. 부산 시군구 / 경계 ----------
P("## 1. 부산 경계·시군구·행정동 (SGIS, 21 필터 → 5186)")
sgg = gpd.read_file(g1("01_행정경계_인구_SGIS행정구역/2. 경계/**/bnd_sigungu*.shp"))
P(f"- 전국 시군구: {len(sgg)}")
bsgg = sgg[sgg.SIGUNGU_CD.str.startswith("21")].copy().to_crs(CRS)
bsgg["sgg_cd"] = bsgg.SIGUNGU_NM.map(SGG)
bsgg["sgg_nm"] = bsgg.SIGUNGU_NM
assert bsgg.sgg_cd.notna().all(), bsgg[bsgg.sgg_cd.isna()].SIGUNGU_NM.tolist()
P(f"- 부산 시군구: {len(bsgg)} (기대 16) {'OK' if len(bsgg)==16 else 'FAIL'}")
bsgg[["sgg_cd","sgg_nm","SIGUNGU_CD","geometry"]].to_file(GRID/"부산_시군구.gpkg", driver="GPKG")

bnd = bsgg.dissolve().reset_index(drop=True)[["geometry"]]
bnd["name"] = "부산광역시"
bnd.to_file(GRID/"부산_경계.gpkg", driver="GPKG")
area_km2 = bnd.geometry.area.iloc[0] / 1e6
P(f"- 부산 경계 dissolve 면적: {area_km2:,.1f} km² (통계청 공식 약 770~772)")
BND = bnd.geometry.iloc[0]

dong = gpd.read_file(g1("01_행정경계_인구_SGIS행정구역/2. 경계/**/bnd_dong*.shp"))
bdong = dong[dong.ADM_CD.str.startswith("21")].copy().to_crs(CRS)
bdong["sgg_cd"] = bdong.ADM_CD.str[:5].map({k:v for k,v in zip(bsgg.SIGUNGU_CD, bsgg.sgg_cd)})
bdong = bdong[["ADM_CD","ADM_NM","sgg_cd","geometry"]]
bdong.to_file(GRID/"부산_행정동.gpkg", driver="GPKG")
P(f"- 부산 행정동: {len(bdong)} (기대 206) {'OK' if len(bdong)==206 else 'FAIL'}, sgg_cd 결측 {bdong.sgg_cd.isna().sum()}")
P()

def clip_report(name, gdf_src_n, gdf_out):
    P(f"- {name}: 원본 {gdf_src_n} → 부산 {len(gdf_out)}  (CRS {gdf_out.crs.to_epsg()})")

# ---------- 2. SGIS 1km 격자 ----------
P("## 2. SGIS 1km 격자 (30블록 병합 → 5186 → 부산 intersects)")
gfiles = sorted(glob.glob(str(RAW/"02_인구_SGIS격자/**/grid_*_1K.shp"), recursive=True))
parts = [gpd.read_file(f)[["GRID_CD","geometry"]] for f in gfiles]
g1k = pd.concat(parts, ignore_index=True)
g1k = gpd.GeoDataFrame(g1k, crs=parts[0].crs).to_crs(CRS)
n0 = len(g1k)
g1k = g1k[g1k.intersects(BND)].reset_index(drop=True)
g1k.to_file(OUT/"sgis_격자1km.gpkg", driver="GPKG")
clip_report("SGIS 1km 격자", n0, g1k)
P()

# ---------- 3. 홍수위험지도 ----------
P("## 3. 홍수위험지도 (해제본 병합, 이미 5186)")
def load_hazard(subdir, tag):
    fs = sorted(glob.glob(str(HAE/"04_위험_홍수위험지도_부산"/subdir/"**"/"RFM_*.shp"), recursive=True)) \
       + sorted(glob.glob(str(HAE/"04_위험_홍수위험지도_부산"/subdir/"**"/"CFM_*.shp"), recursive=True))
    rows = []
    for f in fs:
        gdf = gpd.read_file(f)
        gdf["src_file"] = Path(f).name
        gdf["hazard"] = tag
        rows.append(gdf)
    m = gpd.GeoDataFrame(pd.concat(rows, ignore_index=True), crs=rows[0].crs)
    return m
h_ntn = load_hazard("01_국가하천_하천범람_200년", "국가하천_200")
h_rgn = load_hazard("02_지방하천_하천범람_200년", "지방하천_200")
h_cty = load_hazard("03_도시침수지도_100년", "도시침수_100")
for nm, m in [("국가하천_200", h_ntn), ("지방하천_200", h_rgn), ("도시침수_100", h_cty)]:
    m = m.to_crs(CRS)
    P(f"- {nm}: {len(m)} feature, SEG_CODE 값 {sorted(m.SEG_CODE.dropna().unique().tolist())}, 파일 {m.src_file.nunique()}개")
    m.to_file(OUT/f"홍수위험지도_{nm}.gpkg", driver="GPKG")
P("  ※ SEG_CODE N330~N334 = 침수심 등급 5구간 (추론: N330<0.5m ~ N334≥3.0m, 한강홍수통제소 표준). 속성정의서로 S5 확정")
P()

# ---------- 4. 침수흔적도 (3857 → 5186) ----------
P("## 4. 침수흔적도 (3857 → 5186)")
fmap = g1("05_위험_침수흔적도_부산/침수흔적도_부산.geojson")
fs = gpd.read_file(fmap)
P(f"- 원본 CRS {fs.crs.to_epsg()}, {len(fs)} feature")
fs = fs.to_crs(CRS)
inside = fs.intersects(BND).sum()
P(f"- 부산 경계와 교차: {inside}/{len(fs)}")
fs.to_file(OUT/"침수흔적도.gpkg", driver="GPKG")
P()

# ---------- 5. 포인트 CSV → 5186 ----------
P("## 5. 포인트 레이어 (위경도 4326 → 5186)")
BND_4326 = gpd.GeoSeries([BND], crs=CRS).to_crs(4326).iloc[0]

pmp = pd.read_csv(g1("08_대응_배수펌프장_전국표준/*.csv"), encoding="cp949")
pmp_b = pmp[pmp["시도명"] == "부산광역시"].copy()
pmp_b["경도"] = pd.to_numeric(pmp_b["경도"], errors="coerce")
pmp_b["위도"] = pd.to_numeric(pmp_b["위도"], errors="coerce")
pmp_b = pmp_b.dropna(subset=["경도","위도"])
gp = gpd.GeoDataFrame(pmp_b, geometry=[Point(xy) for xy in zip(pmp_b["경도"], pmp_b["위도"])], crs=4326).to_crs(CRS)
gp.to_file(OUT/"배수펌프장.gpkg", driver="GPKG")
P(f"- 배수펌프장: 전국 {len(pmp)} → 부산 {len(gp)}")

cctv = pd.read_csv(g1("09_결합키_CCTV_부산방범용/*.csv"), encoding="cp949")
cctv["위도"] = pd.to_numeric(cctv["위도"], errors="coerce")
cctv["경도"] = pd.to_numeric(cctv["경도"], errors="coerce")
cctv = cctv.dropna(subset=["위도","경도"])
n_cctv0 = len(cctv)
cctv = cctv[(cctv.위도.between(34.8,35.5)) & (cctv.경도.between(128.7,129.4))]
gc = gpd.GeoDataFrame(cctv, geometry=[Point(xy) for xy in zip(cctv["경도"], cctv["위도"])], crs=4326).to_crs(CRS)
gc.to_file(OUT/"cctv.gpkg", driver="GPKG")
P(f"- CCTV: {n_cctv0} → {len(cctv)} (부산 bbox 필터 후, 제외 {n_cctv0-len(cctv)})")
P()

# ---------- 6. 건물 (이미 5186·부산) ----------
P("## 6. 건물 GIS건물통합정보 — 이미 EPSG:5186·부산 전용, 재투영/클립 불필요")
bshp = g1("03_건물_GIS건물통합정보_부산/**/AL_D010_26_20260809.shp")
P(f"- 원본 사용: {Path(bshp).name} (S5에서 연면적 dasymetric에 직접 사용)")
P()

# ---------- 7. 미처리 (S5로 이월) ----------
P("## 7. S2 범위 밖 (좌표 없음 → S5 지오코딩/코드매핑)")
P("- 지하차도(11): 시설명만 → S5 지오코딩")
P("- 재해위험지구(10): 법정동코드+지번 → S5 법정동 폴리곤 매핑 (피처 아님, 비교용)")
P("- ASOS(07)/AWS(12): 지점 좌표는 KMA 지점메타 → S5 강우 보간")
P()

P("## 검증 (S0 §9)")
P(f"- CRS: 산출물 전부 EPSG:{CRS} ✓")
P(f"- 부산 경계 면적 {area_km2:,.1f} km² (공식치 대비 오차 확인)")
P(f"- 카운트 보존: 시군구 16/16, 행정동 206/206 = {'OK' if len(bsgg)==16 and len(bdong)==206 else 'CHECK'}")
P(f"- 산출 위치: 공공데이터/가공데이터/00_정합_5186/ , 01_격자/")

(REP/"S2_좌표통일.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'S2_좌표통일.md'}")
