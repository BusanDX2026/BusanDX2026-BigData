# -*- coding: utf-8 -*-
"""
S5g. 강우(연최대 일강수량) IDW 보간 → 100m 격자
- 입력: raw/07_.../ASOS_일자료_부산_1990-2026_통합.csv, raw/12_.../AWS_일자료_부산_2005-2026_통합.csv,
        raw/12_.../KMA_지점좌표_ASOS_AWS.csv
- 출력: 02_레이어별/강우_grid.parquet  (grid_id: rain_annmax_mm, rain_ratio)
- 방법(사용자 확정): IDW power=2. 지점별 '평년 연최대 일강수량'(완전연도 평균) → 좌표 4326→5186 → 격자중심 보간
- 완전연도: 2026 제외, 6~9월(주 강우기) 관측 존재하는 연도만
- 규약: S0 §1(5186), §5(강수량 음수→0), 2026 부분연도는 통계 제외(README)
"""
import sys, io, glob, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd

ROOT = Path(__file__).resolve().parents[2]   # 저장소 루트 (절대경로 하드코딩 금지)
RAW = ROOT / "공공데이터" / "raw"
GG = ROOT / "공공데이터" / "가공데이터"
OUT = GG / "02_레이어별"
REP = GG / "_리포트"
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# S5g. 강우 IDW(power=2) → 100m 격자")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

coord = pd.read_csv(RAW / "12_위험_강우_AWS_부산" / "KMA_지점좌표_ASOS_AWS.csv")
coord["STN"] = coord["STN"].astype(str)
P(f"KMA 지점좌표: {len(coord)}개")

def load_daily(path, val_col):
    df = pd.read_csv(path, encoding="utf-8-sig")
    col = [c for c in df.columns if val_col in c][0]
    df = df.rename(columns={col: "rain", "일시": "date", "지점": "stn"})
    df["stn"] = df["stn"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rain"] = pd.to_numeric(df["rain"], errors="coerce").clip(lower=0)
    df = df.dropna(subset=["date"])
    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.month
    return df[["stn", "date", "year", "month", "rain"]]

asos = load_daily(glob.glob(str(RAW / "07_*/ASOS_일자료_부산_*_통합.csv"))[0], "일강수량")
aws = load_daily(glob.glob(str(RAW / "12_*/AWS_일자료_부산_*_통합.csv"))[0], "일강수량")
alld = pd.concat([asos, aws], ignore_index=True)
alld = alld[(alld.year >= 1990) & (alld.year <= 2025)]
P(f"일자료: ASOS {len(asos):,} + AWS {len(aws):,}  (2026 제외 후 {len(alld):,})")

# 지점×연도 연최대, 완전연도(6~9월 관측 존재) 필터
grp = alld.groupby(["stn", "year"])
annmax = grp["rain"].max().rename("annmax")
mon_ok = grp["month"].apply(lambda m: set([6, 7, 8, 9]).issubset(set(m))).rename("mon_ok")
ay = pd.concat([annmax, mon_ok], axis=1).reset_index()
ay = ay[ay.mon_ok]
st = ay.groupby("stn").agg(rain_annmax_mm=("annmax", "mean"), n_year=("year", "nunique")).reset_index()
st = st.merge(coord[["STN", "NAME", "LAT", "LON"]], left_on="stn", right_on="STN", how="inner")
P(f"\n지점별 평년 연최대 일강수량 (완전연도 평균):")
for _, r in st.sort_values("rain_annmax_mm", ascending=False).iterrows():
    P(f"  {r.stn:>4} {r.NAME:<12} {r.rain_annmax_mm:6.1f} mm  (완전연도 {int(r.n_year)})")

# 좌표 4326 → 5186
gpt = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st.LON, st.LAT), crs=4326).to_crs(5186)
sx, sy, sv = gpt.geometry.x.values, gpt.geometry.y.values, gpt.rain_annmax_mm.values

grid = gpd.read_file(GG / "01_격자" / "grid_100m.gpkg")[["grid_id", "x_cen", "y_cen"]]
gx, gy = grid.x_cen.values, grid.y_cen.values

# IDW power=2
def idw(px, py, p=2.0, eps=1.0):
    dx = px[:, None] - sx[None, :]
    dy = py[:, None] - sy[None, :]
    d2 = dx * dx + dy * dy
    d2 = np.maximum(d2, eps)          # 지점과 겹치는 격자 방지
    w = 1.0 / d2**(p / 2)
    return (w @ sv) / w.sum(axis=1)

vals = idw(gx, gy)
mean_all = float(np.mean(vals))
df = pd.DataFrame({
    "grid_id": grid.grid_id.values,
    "rain_annmax_mm": np.round(vals, 2),
    "rain_ratio": np.round(vals / mean_all, 4),
})
df.to_parquet(OUT / "강우_grid.parquet", index=False)

P(f"\n## 검증")
P(f"- 사용 지점 {len(st)}개 (부산권 ASOS+AWS)")
P(f"- 격자 보간값 mm: {df.rain_annmax_mm.describe(percentiles=[.5,.9]).round(1).to_dict()}")
P(f"- 공간비율(부산평균={mean_all:.1f}mm 대비): {df.rain_ratio.min():.2f} ~ {df.rain_ratio.max():.2f}")
P(f"- 방법: IDW p=2, 전 지점 사용. ASOS(재현빈도 크기)는 모델링에서 결합; 여기선 공간패턴만")
P(f"- 산출: {OUT/'강우_grid.parquet'}")

(REP / "S5g_강우.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'S5g_강우.md'}", flush=True)
