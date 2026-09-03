# -*- coding: utf-8 -*-
"""
S5e. 지하차도 지오코딩(브이월드) → 100m 격자
- 입력: raw/11_결합_지하차도_부산/부산_지하차도_현황_*.csv (58행, 시설명·시군구·총길이/폭/높이)
- 출력: 02_레이어별/지하차도_grid.parquet (grid_id: underpass_cnt, underpass_len_m, underpass_min_height_m),
        02_레이어별/지하차도_지오코딩결과.csv (감사용)
- 방법(사용자 확정): 브이월드 검색 API로 '시설명' → 좌표. 실패 시 '부산 시군구 시설명' 재시도.
- 규약: S0 §5(부산 bbox 밖 무시), §8(키는 코드 아님 → secrets/)
- 용도: 노출(지하공간) + 호우 시 선제 통제 대상
"""
import sys, io, glob, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, geopandas as gpd
from _vworld import get_key, search_place, geocode_address, polite

ROOT = Path(__file__).resolve().parents[2]   # 저장소 루트 (절대경로 하드코딩 금지)
RAW = ROOT / "공공데이터" / "raw"
GG = ROOT / "공공데이터" / "가공데이터"
OUT = GG / "02_레이어별"
REP = GG / "_리포트"
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# S5e. 지하차도 지오코딩 → 100m 격자")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")
KEY = get_key()

f = glob.glob(str(RAW / "11_결합_지하차도_부산" / "*.csv"))[0]
d = pd.read_csv(f, encoding="cp949")
d["총길이"] = pd.to_numeric(d["총길이"], errors="coerce")
d["높이"] = pd.to_numeric(d["높이"], errors="coerce")
# 시설명 중복(상·하행) → 최대 총길이 대표
d = d.sort_values("총길이", ascending=False).drop_duplicates(["시설명", "시군구"]).reset_index(drop=True)
P(f"지하차도: {len(d)}건 (시설명×시군구 유일)")

import re
def norm(s):
    s = re.sub(r"\([^)]*\)", "", s)          # 괄호 제거
    s = re.sub(r"\s+", "", s).strip()
    return s

recs = []
for _, r in d.iterrows():
    nm, gu = str(r["시설명"]).strip(), str(r["시군구"]).strip()
    nn = norm(nm)
    base = re.sub(r"(지하차도|지하도|지하보차도|굴다리|복개구조물|생태통로)$", "", nn).strip()
    tries = [
        ("search:시설명", lambda: search_place(nm, KEY)),
        ("search:norm", lambda: search_place(nn, KEY)),
        ("search:구+norm", lambda: search_place(f"부산 {gu} {nn}", KEY)),
        ("search:구+base+교차로", lambda: search_place(f"부산 {gu} {base}교차로", KEY) if base else None),
        ("search:구+base+사거리", lambda: search_place(f"부산 {gu} {base}사거리", KEY) if base else None),
        ("road", lambda: geocode_address(f"부산광역시 {gu} {nn}", KEY, kind="road")),
        ("search:구+base", lambda: search_place(f"부산 {gu} {base}", KEY) if base and len(base) >= 2 else None),
    ]
    pt, how = None, "FAIL"
    for label, fn in tries:
        pt = fn(); polite()
        if pt:
            how = label; break
    lon, lat = (pt if pt else (np.nan, np.nan))
    recs.append(dict(시설명=nm, 시군구=gu, 총길이=r["총길이"], 높이=r["높이"], lon=lon, lat=lat, method=how))
res = pd.DataFrame(recs)
ok = res.dropna(subset=["lon"])
P(f"- 지오코딩 성공 {len(ok)}/{len(res)}  (실패 {len(res)-len(ok)})")
for _, r in res[res.lon.isna()].iterrows():
    P(f"    실패: {r.시군구} {r.시설명}")
res.to_csv(OUT / "지하차도_지오코딩결과.csv", index=False, encoding="utf-8-sig")

# 격자 매핑
gridf = gpd.read_file(GG / "01_격자" / "grid_100m.gpkg")[["grid_id", "x_cen", "y_cen", "geometry"]].to_crs(5186)
grid = gridf[["grid_id", "geometry"]]
gp = gpd.GeoDataFrame(ok, geometry=gpd.points_from_xy(ok.lon, ok.lat), crs=4326).to_crs(5186)
j = gpd.sjoin(gp, grid, how="left", predicate="within").dropna(subset=["grid_id"])
agg = j.groupby("grid_id").agg(
    underpass_cnt=("시설명", "nunique"),
    underpass_len_m=("총길이", "sum"),
    underpass_min_height_m=("높이", "min"),
).reset_index()
full = gridf[["grid_id", "x_cen", "y_cen"]].merge(agg, on="grid_id", how="left")
for c in ["underpass_cnt", "underpass_len_m"]:
    full[c] = full[c].fillna(0)

# 단일 격자 지하차도는 초희소(전체의 0.06%) → 모델 입력용 이웃 반경 피처 추가
from scipy.spatial import cKDTree
uxy = np.c_[gp.geometry.x.values, gp.geometry.y.values]
gxy = np.c_[full.x_cen.values, full.y_cen.values]
tree = cKDTree(uxy)
full["underpass_n_300m"] = tree.query_ball_point(gxy, r=300, return_length=True).astype(int)
full["underpass_n_500m"] = tree.query_ball_point(gxy, r=500, return_length=True).astype(int)
full["underpass_dist_m"] = np.round(tree.query(gxy, k=1)[0], 1)
full = full.drop(columns=["x_cen", "y_cen"])
full.to_parquet(OUT / "지하차도_grid.parquet", index=False)

P(f"\n## 검증")
P(f"- 격자 매핑 성공 {len(j)} 지점 → {int((full.underpass_cnt>0).sum())} 격자 (전체의 {(full.underpass_cnt>0).mean():.3%} — 초희소)")
P(f"- 이웃 확장: 300m내 {int((full.underpass_n_300m>0).sum()):,}격자 / 500m내 {int((full.underpass_n_500m>0).sum()):,}격자 "
  f"→ 모델 입력은 이웃 피처 사용 권장")
P(f"- 구별 지하차도 수(지오코딩된 것): {ok.assign(g=ok.시군구).groupby('g').size().to_dict()}")
P(f"- 산출: {OUT/'지하차도_grid.parquet'} (컬럼 underpass_cnt/len_m/min_height_m), {OUT/'지하차도_지오코딩결과.csv'}")

(REP / "S5e_지하차도.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'S5e_지하차도.md'}", flush=True)
