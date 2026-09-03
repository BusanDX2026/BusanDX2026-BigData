# -*- coding: utf-8 -*-
"""
D1. 환경부(기후에너지환경부) 토지피복지도 세분류 — WFS 수집 (부산)

왜 필요한가
  현재 모델의 최대 약점은 내수침수 근거 부재다(전처리 한계 L-2, 모델 이슈 #5).
  불투수면적은 내수침수 유출량의 직접 원인이면서, 동시에
  "강서 삼각주는 저지대지만 농경지" 를 모델이 구분하게 해 이슈 #5(역방향 학습)를 함께 푼다.

출처
  기후에너지환경부 환경공간정보서비스 GeoServer WFS (인증키 불필요, 공개)
  https://api.mcee.go.kr/geoserver/wfs
  레이어: EGIS:landcover_lv3_11th (세분류 토지피복지도 11차, 2021)
  속성: l1_code/l1_name(대분류) · l2_*(중분류) · l3_*(세분류) · img_date

- 출력: 공공데이터/raw/13_토지피복_세분류_부산/토지피복_세분류_부산.gpkg
- 규약: S0 §1 (EPSG:5186), §8 (인증키 없음 — 공개 WFS)
"""
import sys, io, json, time, urllib.parse, urllib.request, hashlib, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import geopandas as gpd, pandas as pd
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "공공데이터" / "raw" / "13_토지피복_세분류_부산"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "_tiles"; CACHE.mkdir(exist_ok=True)

BASE = "https://api.mcee.go.kr/geoserver/wfs"
LAYER = "EGIS:landcover_lv3_11th"
UA = {"User-Agent": "Mozilla/5.0 (academic research; busan flood analysis)"}
# 부산 bbox (EPSG:5186) — S4 격자 범위와 동일
X0, Y0, X1, Y1 = 360700, 256000, 409700, 312500
STEP = 5000            # 5km 타일
PAGE = 20000           # 타일당 최대 (초과 시 경고)

def fetch(bbox, retry=3):
    q = urllib.parse.urlencode({
        "service": "WFS", "version": "1.1.0", "request": "GetFeature",
        "typeName": LAYER, "srsName": "EPSG:5186",
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},EPSG:5186",
        "outputFormat": "application/json", "maxFeatures": str(PAGE),
    })
    url = f"{BASE}?{q}"
    for k in range(retry):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            if k == retry - 1:
                raise
            time.sleep(3 * (k + 1))

tiles = [(x, y, min(x + STEP, X1), min(y + STEP, Y1))
         for x in range(X0, X1, STEP) for y in range(Y0, Y1, STEP)]
print(f"타일 {len(tiles)}개 ({STEP/1000:.0f}km) · 레이어 {LAYER}")

t0 = time.perf_counter()
for i, bb in enumerate(tiles, 1):
    cf = CACHE / f"t_{bb[0]}_{bb[1]}.json"
    if cf.exists():
        continue
    j = fetch(bb)
    n = len(j.get("features", []))
    cf.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
    if n >= PAGE:
        print(f"  ⚠ 타일 {bb[:2]} 상한 도달({n}) — STEP 축소 필요")
    print(f"  [{i}/{len(tiles)}] {bb[0]},{bb[1]} → {n:,}건  ({time.perf_counter()-t0:.0f}s)")
    time.sleep(0.2)

print("\n타일 병합 중…")
seen, recs, geoms = set(), [], []
KEEP = ["l1_code", "l1_name", "l2_code", "l2_name", "l3_code", "l3_name", "img_date"]
for cf in sorted(CACHE.glob("t_*.json")):
    j = json.loads(cf.read_text(encoding="utf-8"))
    for f in j.get("features", []):
        g = f.get("geometry")
        if not g:
            continue
        h = hashlib.md5(json.dumps(g, sort_keys=True).encode()).hexdigest()
        if h in seen:               # 타일 경계 중복 제거
            continue
        seen.add(h)
        p = f.get("properties", {})
        recs.append({k: p.get(k) for k in KEEP})
        geoms.append(shape(g))
gdf = gpd.GeoDataFrame(pd.DataFrame(recs), geometry=geoms, crs=5186)
print(f"병합 결과: {len(gdf):,} 폴리곤 (중복 제거 후)")
print(f"대분류 분포:\n{gdf.l1_name.value_counts().to_string()}")
gdf["area_m2"] = gdf.area
print(f"\n총 면적 {gdf.area_m2.sum()/1e6:,.1f} km² (부산 785.6 km² 대비 — bbox 기준이라 더 넓음)")
dst = OUT / "토지피복_세분류_부산.gpkg"
gdf.to_file(dst, driver="GPKG")
print(f"저장: {dst}")

readme = OUT / "README.md"
readme.write_text(f"""# 13. 토지피복지도 세분류 (부산)

## 데이터 특성
- **형식**: GeoPackage `토지피복_세분류_부산.gpkg` — **{len(gdf):,} 폴리곤**, EPSG:5186
- **분류**: 대분류 `l1_*` (7종) → 중분류 `l2_*` (22종) → **세분류 `l3_*` (41종)**
- **기준영상**: 항공정사영상, `img_date` 컬럼 참조
- **레이어**: `EGIS:landcover_lv3_11th` (세분류 토지피복지도 11차, 2021)
- **수집 범위**: 부산 bbox EPSG:5186 ({X0},{Y0})~({X1},{Y1}), {STEP/1000:.0f}km 타일 {len(tiles)}개 분할 수집 후 중복 제거

## 왜 필요한가
- **내수침수의 직접 원인**: 불투수면적률(시가화건조지역 비율)이 높을수록 유출량↑ → 내수침수 위험↑.
  기존 데이터엔 이 변수가 없어 지형만으로 배수를 추정하고 있었음 (전처리 한계 L-2)
- **모델 이슈 #5 해결**: 모델이 강서 삼각주 저지대를 "안전"으로 역방향 학습한 원인은
  '저지대인데 농경지라 침수기록이 없다'는 점을 구분하지 못해서였음.
  토지피복이 있으면 "저지대+시가지"와 "저지대+농경지"를 구분 가능
- 도시침수지도가 강서구만 구축된 공백을 부분적으로 보완

## 대분류 분포
```
{gdf.l1_name.value_counts().to_string()}
```

## 출처
- 제공: 기후에너지환경부 환경공간정보서비스 (구 환경부 EGIS)
- WFS: `https://api.mcee.go.kr/geoserver/wfs` (**인증키 불필요, 공개 서비스**)
- 화면: https://aid.mcee.go.kr — 자료신청은 로그인 필요하나 WFS는 공개
- 라이선스: 공공누리 (출처표시)
- 수집일: {time.strftime('%Y-%m-%d')}
""", encoding="utf-8")
print(f"README: {readme}")
