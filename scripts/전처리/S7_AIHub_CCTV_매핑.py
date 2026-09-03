# -*- coding: utf-8 -*-
"""
S7. AI-Hub 「침수위험 지역 라벨링 데이터」 측구(gutter) 관측지점 ↔ 09 부산 방범 CCTV 매핑
- 입력: AI-Hub 「부산시 침수위험 복합 데이터」 라벨링 JSON (Training+Validation 90,000개) —
          info.gutter_position_lat / gutter_position_lon 을 스캔해 관측지점 추출
        00_정합_5186/cctv.gpkg (부산 방범 CCTV), 01_격자/grid_100m.gpkg
- 출력: 02_레이어별/aihub_관측지점_원시.csv (스캔 캐시, 재실행 시 재사용)
        02_레이어별/aihub_cctv_매핑.csv
- 용도: 시범구역 사례분석·검증에서 AI-Hub 이미지 지점을 격자/행정동에 연결. (피처 아님)
- 규약: S0 §5 (부산 bbox 밖 좌표 무시), §8 (재현성 — 저장소 내부 경로만 사용)
"""
import sys, io, json, collections
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, geopandas as gpd
from scipy.spatial import cKDTree
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"
OUT = GG / "02_레이어별"; OUT.mkdir(parents=True, exist_ok=True)
REP = GG / "_리포트"
AIHUB = ROOT / "부산시 침수위험 복합 데이터"
SITES_CSV = OUT / "aihub_관측지점_원시.csv"
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# S7. AI-Hub 관측지점 ↔ 부산 방범 CCTV 매핑")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

# --- AI-Hub 라벨링 JSON 스캔 (캐시 있으면 재사용) ---
if SITES_CSV.exists():
    sites = pd.read_csv(SITES_CSV)
    P(f"관측지점 캐시 사용: {SITES_CSV.name} ({len(sites)}행)")
else:
    files = list(AIHUB.glob("**/침수위험 지역 라벨링 데이터/**/*.json"))
    if not files:
        raise FileNotFoundError(
            f"AI-Hub 라벨링 JSON 없음: {AIHUB}\n"
            "  → AI-Hub 「부산시 침수위험 복합 데이터」를 내려받아 저장소 루트에 두거나,\n"
            f"     이전 스캔 결과 {SITES_CSV.name} 를 배치하세요.")
    P(f"AI-Hub 라벨링 JSON {len(files):,}개 스캔 중… (수 분 소요, 결과는 캐시됨)")
    cnt = collections.Counter()
    for i, f in enumerate(files):
        try:
            info = json.loads(f.read_text(encoding="utf-8")).get("info", {})
        except Exception:
            continue
        cnt[(round(info.get("gutter_position_lat") or 0, 6),
             round(info.get("gutter_position_lon") or 0, 6))] += 1
        if i and i % 20000 == 0:
            P(f"  .. {i:,}/{len(files):,}")
    sites = pd.DataFrame([(lat, lon, c) for (lat, lon), c in cnt.most_common()],
                         columns=["lat", "lon", "img_cnt"])
    sites.to_csv(SITES_CSV, index=False, encoding="utf-8-sig")
    P(f"스캔 완료 → 캐시 저장 {SITES_CSV.name}")
P(f"gutter 지점 후보: {len(sites)} (원시 스캔)")
sites = sites[(sites.lat != 0) & (sites.lon != 0)]
inb = sites[(sites.lon.between(128.7, 129.4)) & (sites.lat.between(34.9, 35.4))].copy()
oob = sites[~sites.index.isin(inb.index)]
P(f"- (0,0) 결측(외수 폴더, 좌표 없음) 제외 + 부산 bbox 밖 {len(oob)}개 제외 → 유효 지점 {len(inb)}")
for _, r in oob.iterrows():
    P(f"    제외 좌표오류: {r.lat:.5f}, {r.lon:.5f} (img {int(r.img_cnt)})")

# 근접(<80m) 지점은 동일 물리지점으로 병합
inb = inb.sort_values("img_cnt", ascending=False).reset_index(drop=True)
g = gpd.GeoDataFrame(inb, geometry=gpd.points_from_xy(inb.lon, inb.lat), crs=4326).to_crs(5186)
xy = np.c_[g.geometry.x, g.geometry.y]
keep = []
for i in range(len(g)):
    if all(np.hypot(xy[i,0]-xy[j,0], xy[i,1]-xy[j,1]) > 80 for j in keep):
        keep.append(i)
merged = g.iloc[keep].reset_index(drop=True)
P(f"- 80m 이내 중복 병합 → 최종 관측지점 {len(merged)}개\n")

# --- 방범 CCTV / 격자 매핑 ---
cctv = gpd.read_file(GG / "00_정합_5186" / "cctv.gpkg").to_crs(5186)
grid = gpd.read_file(GG / "01_격자" / "grid_100m.gpkg").to_crs(5186)

cxy = np.c_[cctv.geometry.x, cctv.geometry.y]
tree = cKDTree(cxy)
mxy = np.c_[merged.geometry.x, merged.geometry.y]
dist, idx = tree.query(mxy, k=1)
merged["nearest_cctv_dist_m"] = np.round(dist, 1)
namecol = "시설명칭" if "시설명칭" in cctv.columns else cctv.columns[0]
gucol = "구군" if "구군" in cctv.columns else None
merged["nearest_cctv_name"] = cctv.iloc[idx][namecol].values
if gucol:
    merged["nearest_cctv_gu"] = cctv.iloc[idx][gucol].values

j = gpd.sjoin(merged, grid[["grid_id", "sgg_nm", "adm_nm", "geometry"]], how="left", predicate="within")
j = j.drop(columns=[c for c in ["index_right", "geometry"] if c in j.columns])
j["lat"] = merged.to_crs(4326).geometry.y.values
j["lon"] = merged.to_crs(4326).geometry.x.values
cols = ["lat", "lon", "img_cnt", "grid_id", "sgg_nm", "adm_nm",
        "nearest_cctv_name", "nearest_cctv_dist_m"] + (["nearest_cctv_gu"] if gucol else [])
j[cols].to_csv(OUT / "aihub_cctv_매핑.csv", index=False, encoding="utf-8-sig")

P("## 매핑 결과 (AI-Hub 내수 이미지 관측지점 → 격자/CCTV)")
for _, r in j[cols].iterrows():
    P(f"  ({r.lat:.5f},{r.lon:.5f}) img~{int(r.img_cnt):>5} | {r.sgg_nm} {r.adm_nm} | grid {r.grid_id} "
      f"| 최근접CCTV {r.nearest_cctv_dist_m}m '{str(r.nearest_cctv_name)[:30]}'")
P(f"\n- 행정동 분포: {j.adm_nm.value_counts().to_dict()}")
P(f"- 최근접 CCTV 거리(m): min {j.nearest_cctv_dist_m.min()}, median {j.nearest_cctv_dist_m.median()}, max {j.nearest_cctv_dist_m.max()}")
P(f"- 산출: {OUT/'aihub_cctv_매핑.csv'}")
P("\n※ AI-Hub 내수(danger_level=0) 이미지는 소수 지점(온천천 세병교 일대 집중)에 편중. "
  "citywide 피처가 아니라 시범구역 사례분석·모델 검증 앵커로 사용.")

(REP / "S7_AIHub_CCTV.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'S7_AIHub_CCTV.md'}", flush=True)
