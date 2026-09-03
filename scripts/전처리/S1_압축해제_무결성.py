
# -*- coding: utf-8 -*-
"""
S1. 압축 해제 + 무결성 검증
- 입력: 공공데이터/raw/ (zip 포함)
- 출력: 공공데이터/가공데이터/_해제/ (zip 해제본), 공공데이터/가공데이터/_리포트/S1_무결성.md
- 검증: zip 손상 0, SHP 컴포넌트 완비(.shp/.shx/.dbf/.prj), feature 수 > 0, CSV 파싱 가능
- 규약: 문서/S0_작업규약.md
"""
import os, sys, io, zipfile, glob, csv, json, struct, warnings
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]   # 저장소 루트 (절대경로 하드코딩 금지)
RAW = ROOT / "공공데이터" / "raw"
GAGONG = ROOT / "공공데이터" / "가공데이터"
HAECHE = GAGONG / "_해제"
REPORT = GAGONG / "_리포트"

for d in [GAGONG, HAECHE, REPORT, GAGONG/"01_격자", GAGONG/"02_레이어별", GAGONG/"03_마스터"]:
    d.mkdir(parents=True, exist_ok=True)

log = []
problems = []
def P(s=""):
    print(s); log.append(str(s))

P("# S1. 압축 해제 + 무결성 검증")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}")
P()

# ---------- 1. zip 해제 ----------
P("## 1. zip 해제 (홍수위험지도)")
zips = sorted(RAW.glob("**/*.zip"))
P(f"zip 파일: {len(zips)}개")
for z in zips:
    rel = z.relative_to(RAW)
    out = HAECHE / rel.parent / z.stem
    try:
        with zipfile.ZipFile(z) as zf:
            bad = zf.testzip()
            if bad:
                problems.append(f"[손상] {rel} -> {bad}")
                P(f"  ✗ {rel}: 손상 ({bad})")
                continue
            names = zf.namelist()
            out.mkdir(parents=True, exist_ok=True)
            zf.extractall(out)
        exts = sorted({os.path.splitext(n)[1].lower() for n in names if os.path.splitext(n)[1]})
        P(f"  ✓ {rel.name}: {len(names)}개 {exts} -> _해제/{rel.parent}/{z.stem}/")
    except Exception as e:
        problems.append(f"[해제실패] {rel}: {e}")
        P(f"  ✗ {rel}: {e}")
P()

# ---------- 2. SHP 무결성 ----------
P("## 2. SHP 무결성 (raw + 해제본)")
import geopandas as gpd

shp_all = sorted(list(RAW.glob("**/*.shp")) + list(HAECHE.glob("**/*.shp")))
P(f"SHP: {len(shp_all)}개")
P()
P("| 파일 | feature | geom | CRS(EPSG) | .prj | .dbf | bbox(원CRS) |")
P("|---|---|---|---|---|---|---|")
shp_summary = []
for shp in shp_all:
    stem = shp.with_suffix("")
    has_prj = stem.with_suffix(".prj").exists()
    has_dbf = stem.with_suffix(".dbf").exists()
    has_shx = stem.with_suffix(".shx").exists()
    name = str(shp.relative_to(ROOT))
    try:
        gdf = gpd.read_file(shp)
        epsg = gdf.crs.to_epsg() if gdf.crs else None
        gtypes = ",".join(sorted(gdf.geom_type.dropna().unique()))
        b = gdf.total_bounds
        bbox = f"{b[0]:.0f},{b[1]:.0f} ~ {b[2]:.0f},{b[3]:.0f}"
        P(f"| {shp.name} | {len(gdf):,} | {gtypes} | {epsg} | {'O' if has_prj else '✗'} | {'O' if has_dbf else '✗'} | {bbox} |")
        shp_summary.append(dict(file=name, n=len(gdf), geom=gtypes, epsg=epsg, bbox=bbox))
        if len(gdf) == 0:
            problems.append(f"[빈 SHP] {name}")
        if not has_prj or epsg is None:
            problems.append(f"[CRS불명] {name} (.prj={has_prj}, epsg={epsg})")
        if not (has_dbf and has_shx):
            problems.append(f"[컴포넌트누락] {name} (.dbf={has_dbf}, .shx={has_shx})")
    except Exception as e:
        P(f"| {shp.name} | ERROR | | | {'O' if has_prj else '✗'} | {'O' if has_dbf else '✗'} | {e} |")
        problems.append(f"[SHP읽기실패] {name}: {e}")
P()

# ---------- 3. GeoTIFF 무결성 (DEM) ----------
P("## 3. GeoTIFF 무결성 (DEM)")
import rasterio
tifs = sorted(RAW.glob("**/*.tif")) + sorted(RAW.glob("**/*.img"))
P(f"래스터: {len(tifs)}개")
P()
P("| 파일 | 크기(px) | 해상도 | CRS(EPSG) | nodata | bounds |")
P("|---|---|---|---|---|---|")
for t in tifs:
    name = str(t.relative_to(ROOT))
    try:
        with rasterio.open(t) as ds:
            epsg = ds.crs.to_epsg() if ds.crs else None
            res = ds.res
            b = ds.bounds
            P(f"| {t.name} | {ds.width}x{ds.height} | {res[0]:.4g} | {epsg} | {ds.nodata} | {b.left:.3f},{b.bottom:.3f} ~ {b.right:.3f},{b.top:.3f} |")
            if epsg is None:
                problems.append(f"[CRS불명] {name}")
    except Exception as e:
        P(f"| {t.name} | ERROR | | | | {e} |")
        problems.append(f"[래스터읽기실패] {name}: {e}")
P()

# ---------- 4. CSV / JSON 파싱 ----------
P("## 4. CSV / GeoJSON / JSON 파싱")
data_files = []
for pat in ("**/*.csv", "**/*.geojson", "**/*.json"):
    data_files += sorted(RAW.glob(pat))
# SGIS 통계 CSV는 매우 많으므로 대표만
P("| 파일 | 행수 | 인코딩/형식 | 컬럼(앞부분) |")
P("|---|---|---|---|")
shown = 0
for f in data_files:
    name = str(f.relative_to(ROOT))
    # SGIS 통계 폴더는 요약만
    if "1. 통계" in name and shown > 6 and ("SGIS" in name or "격자" in name):
        continue
    try:
        if f.suffix == ".csv":
            enc_ok = None
            for enc in ("utf-8-sig", "cp949", "utf-8"):
                try:
                    with open(f, encoding=enc) as fh:
                        rd = csv.reader(fh); header = next(rd)
                        n = sum(1 for _ in rd)
                    enc_ok = enc; break
                except Exception:
                    continue
            if enc_ok is None:
                problems.append(f"[CSV파싱실패] {name}")
                P(f"| {f.name} | ERROR | 인코딩불명 | |")
            else:
                P(f"| {f.name} | {n:,} | csv/{enc_ok} | {header[:6]} |")
        elif f.suffix in (".geojson", ".json"):
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
            if isinstance(d, dict) and d.get("type") == "FeatureCollection":
                feats = d.get("features", [])
                crs = (d.get("crs") or {}).get("properties", {}).get("name", "?")
                P(f"| {f.name} | {len(feats):,} | geojson | CRS={crs} |")
            elif isinstance(d, list):
                P(f"| {f.name} | {len(d):,} | json(list) | keys={list(d[0].keys())[:5] if d else []} |")
            else:
                P(f"| {f.name} | - | json({type(d).__name__}) | {list(d.keys())[:5] if isinstance(d,dict) else ''} |")
        shown += 1
    except Exception as e:
        problems.append(f"[파싱실패] {name}: {e}")
        P(f"| {f.name} | ERROR | | {e} |")
P()

# ---------- 요약 ----------
P("## 검증 결과")
if problems:
    P(f"### ⚠️ 문제 {len(problems)}건")
    for p in problems:
        P(f"- {p}")
else:
    P("### ✓ 문제 없음 — 모든 zip 해제 성공, SHP/래스터/CSV 무결성 통과")
P()
P(f"- SHP: {len(shp_all)}개 검사")
P(f"- 래스터: {len(tifs)}개 검사")
P(f"- 해제본 위치: `공공데이터/가공데이터/_해제/`")

(REPORT / "S1_무결성.md").write_text("\n".join(log), encoding="utf-8")
print()
print(f"==> 리포트 저장: {REPORT/'S1_무결성.md'}")
print(f"==> 문제: {len(problems)}건")
sys.exit(1 if problems else 0)
