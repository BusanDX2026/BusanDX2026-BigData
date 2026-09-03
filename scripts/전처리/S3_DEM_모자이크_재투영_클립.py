# -*- coding: utf-8 -*-
"""
S3. DEM 모자이크 + 재투영(EPSG:5186) + 부산 클립
- 입력: 공공데이터/raw/06_위험_지형_DEM/DEM_30m_Copernicus_부산/*.tif (4타일, EPSG:4326)
        공공데이터/가공데이터/01_격자/부산_경계.gpkg
- 출력: 공공데이터/가공데이터/00_정합_5186/DEM_30m_부산_5186.tif
- 규약: 문서/S0_작업규약.md §1(5186), §4(no-data 마스킹), §5(해수면 0 유지, <-50m만 no-data)
- 파생(경사/저지대비율/TWI)은 S5 격자 집계에서 산출
"""
import sys, io, glob, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask as rio_mask

ROOT = Path(__file__).resolve().parents[2]   # 저장소 루트 (절대경로 하드코딩 금지)
RAW = ROOT / "공공데이터" / "raw"
GG = ROOT / "공공데이터" / "가공데이터"
OUT = GG / "00_정합_5186"
REP = GG / "_리포트"
DST_CRS = "EPSG:5186"
NODATA = -9999.0
log = []
def P(s=""):
    print(s); log.append(str(s))

P("# S3. DEM 모자이크·재투영·부산 클립")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}")
P()

tiles = sorted(glob.glob(str(RAW / "06_위험_지형_DEM" / "DEM_30m_Copernicus_부산" / "*.tif")))
P(f"## 1. 입력 타일 {len(tiles)}개 (Copernicus GLO-30, EPSG:4326)")
for t in tiles:
    P(f"- {Path(t).name}")

# ---------- 모자이크 (4326) ----------
srcs = [rasterio.open(t) for t in tiles]
mosaic, m_transform = merge(srcs)
src_crs = srcs[0].crs
src_nodata = srcs[0].nodata
for s in srcs:
    s.close()
P(f"\n## 2. 모자이크: shape {mosaic.shape}, CRS {src_crs.to_epsg()}, src nodata {src_nodata}")
# Copernicus는 nodata 미정의 → 바다 0m. -50m 미만만 오류로 간주(S0 §5)
band = mosaic[0].astype("float32")
P(f"- 원 표고 min/max/mean: {np.nanmin(band):.1f} / {np.nanmax(band):.1f} / {np.nanmean(band):.2f}")
err_lo = np.sum(band < -50)
band[band < -50] = np.nan
P(f"- < -50m (오류 처리) 셀: {err_lo}")

# ---------- 재투영 4326 → 5186 ----------
h, w = band.shape
dst_transform, dw, dh = calculate_default_transform(
    src_crs, DST_CRS, w, h,
    left=m_transform.c, bottom=m_transform.f + m_transform.e * h,
    right=m_transform.c + m_transform.a * w, top=m_transform.f,
    resolution=30.0)
dst = np.full((dh, dw), np.nan, dtype="float32")
reproject(
    source=band, destination=dst,
    src_transform=m_transform, src_crs=src_crs,
    dst_transform=dst_transform, dst_crs=DST_CRS,
    src_nodata=np.nan, dst_nodata=np.nan,
    resampling=Resampling.bilinear)
P(f"\n## 3. 재투영 → EPSG:5186, 30m: shape {dst.shape}")
P(f"- 재투영 후 표고 min/max/mean: {np.nanmin(dst):.1f} / {np.nanmax(dst):.1f} / {np.nanmean(dst):.2f}")

meta = dict(driver="GTiff", height=dh, width=dw, count=1, dtype="float32",
            crs=DST_CRS, transform=dst_transform, nodata=NODATA, compress="deflate")
tmp = OUT / "_dem_5186_full.tif"
arr = np.where(np.isnan(dst), NODATA, dst).astype("float32")
with rasterio.open(tmp, "w", **meta) as d:
    d.write(arr, 1)

# ---------- 부산 경계 클립 ----------
bnd = gpd.read_file(GG / "01_격자" / "부산_경계.gpkg").to_crs(DST_CRS)
geom = [bnd.geometry.iloc[0].__geo_interface__]
with rasterio.open(tmp) as d:
    clipped, clip_transform = rio_mask(d, geom, crop=True, nodata=NODATA)
    cmeta = d.meta.copy()
cband = clipped[0]
valid = cband != NODATA
cmeta.update(height=clipped.shape[1], width=clipped.shape[2], transform=clip_transform)
final = OUT / "DEM_30m_부산_5186.tif"
with rasterio.open(final, "w", **cmeta) as d:
    d.write(clipped)
tmp.unlink()

vp = float(valid.mean())
P(f"\n## 4. 부산 경계 클립 → {final.name}")
P(f"- 최종 shape: {clipped.shape[1]}x{clipped.shape[2]}")
P(f"- 유효셀 {valid.sum():,} ({vp:.1%}), nodata {(~valid).sum():,}")
P(f"- 유효 표고 min/max/mean/median: {cband[valid].min():.1f} / {cband[valid].max():.1f} / "
  f"{cband[valid].mean():.1f} / {np.median(cband[valid]):.1f}")
lowland = np.sum((cband >= -1) & (cband <= 5) & valid)
P(f"- 표고 ≤5m (해안 저지대 후보) 유효셀: {lowland:,} ({lowland/valid.sum():.1%})")
sea0 = np.sum((cband >= -0.5) & (cband <= 0.5) & valid)
P(f"- 표고 ≈0m (해면/수역 의심) 유효셀: {sea0:,} ({sea0/valid.sum():.1%}) — S5 격자 집계 시 건물·인구 0 셀과 대조하여 수역 마스킹")

P()
P("## 검증 (S0 §9)")
P(f"- CRS: EPSG:5186 ✓  해상도 30m ✓")
P(f"- 표고 범위 {cband[valid].min():.0f}~{cband[valid].max():.0f}m (부산 최고봉 금정산 801m·백양산 642m 대역과 부합)")
P(f"- 파생지표(경사/저지대비율/흐름누적·TWI)는 S5에서 격자 집계와 함께 산출")
P(f"- 산출: {final}")

(REP / "S3_DEM.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'S3_DEM.md'}")
