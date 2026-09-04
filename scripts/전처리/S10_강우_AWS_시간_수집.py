# -*- coding: utf-8 -*-
"""
S10. AWS/ASOS 시간강우 수집 → 지점×시간 long 테이블

목적
  침수흔적 폴리곤별 '유발 강우강도'를 붙이기 위한 기반 데이터.
  기존 강우 피처(rain_annmax_mm)는 연최대일강수 IDW라 부산 내 공간분산이 std 5.4mm로
  사실상 상수 = 공간지문. 시나리오(활성화 강우) 모델의 축을 관측 유발강우로 바꾸려면
  사건 시각 단위 강우가 필요하다.

입력
  공공데이터/raw/방재기상관측/**/SURFACE_AWS_{stn}_HR_*.zip      (15개 지점, 2009~2025, 시간)
  공공데이터/raw/종관기상관측(ASOS)/SURFACE_ASOS_{stn}_HR_*.zip   (159 부산, 296 북부산)
  공공데이터/raw/12_위험_강우_AWS_부산/KMA_지점좌표_ASOS_AWS.csv   (16지점 위경도)
출력
  02_레이어별/강우_AWS시간_long.parquet   [stn, tm, rain_mm, src]
  02_레이어별/강우_AWS_지점.csv           [stn, name, lat, lon, alt_m, type, n_hours, yr_min, yr_max]
  _리포트/S10_강우AWS시간.md
규약: S0 §4(결측 유지·기록), 인증키/개인정보 코드 미포함
"""
import sys, io, zipfile, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "공공데이터" / "raw"
OUT = ROOT / "공공데이터" / "가공데이터" / "02_레이어별"
REP = ROOT / "공공데이터" / "가공데이터" / "_리포트"
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# S10. AWS/ASOS 시간강우 수집")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

zips = sorted(RAW.glob("방재기상관측/**/SURFACE_AWS_*_HR_*.zip")) + \
       sorted(RAW.glob("종관기상관측(ASOS)/**/SURFACE_ASOS_*_HR_*.zip")) + \
       sorted(RAW.glob("종관기상관측(ASOS)/SURFACE_ASOS_*_HR_*.zip"))
zips = sorted(set(zips))
P(f"대상 zip: {len(zips)}개")

frames = []
bad = []
for zp in zips:
    src = "ASOS" if "ASOS" in zp.name else "AWS"
    try:
        with zipfile.ZipFile(zp) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not names:
                bad.append((zp.name, "csv 없음")); continue
            with zf.open(names[0]) as fh:
                raw = fh.read()
        txt = raw.decode("cp949", errors="replace")
        df = pd.read_csv(io.StringIO(txt))
    except Exception as e:
        bad.append((zp.name, str(e)[:60])); continue

    cols = list(df.columns)
    stn_c = cols[0]           # '지점'
    tm_c = cols[1]            # '일시'
    rain_c = next((c for c in cols if "강수량" in str(c)), None)
    if rain_c is None:
        bad.append((zp.name, "강수량 컬럼 없음")); continue

    sub = df[[stn_c, tm_c, rain_c]].copy()
    sub.columns = ["stn", "tm", "rain_mm"]
    sub["stn"] = pd.to_numeric(sub["stn"], errors="coerce").astype("Int64")
    sub["tm"] = pd.to_datetime(sub["tm"], errors="coerce")
    sub["rain_mm"] = pd.to_numeric(sub["rain_mm"], errors="coerce")
    sub["src"] = src
    sub = sub.dropna(subset=["stn", "tm"])
    frames.append(sub)

rain = pd.concat(frames, ignore_index=True)
n_raw = len(rain)
# 중복(다중 다운로드 배치) 제거 — 같은 지점·시각은 강수량 큰 쪽(관측 우선) 유지
rain = (rain.sort_values(["stn", "tm", "rain_mm"], na_position="first")
             .drop_duplicates(["stn", "tm"], keep="last")
             .reset_index(drop=True))
P(f"- 원시행 {n_raw:,} → 중복제거 {len(rain):,}  (지점 {rain.stn.nunique()}개, "
  f"{rain.tm.min():%Y-%m} ~ {rain.tm.max():%Y-%m})")

# 결측 처리: 강수량 공란은 KMA 관례상 '강수 없음(0)' — 다만 비율 기록
n_na = int(rain.rain_mm.isna().sum())
P(f"- rain_mm 결측 {n_na:,} ({n_na/len(rain):.2%}) → 0.0 대체 (KMA 시간자료: 공란=무강수)")
rain["rain_mm"] = rain["rain_mm"].fillna(0.0).clip(lower=0)
# 물리 상한 점검 (시간 150mm 초과는 국내 관측 극값권 → 값은 유지하고 로그만)
n_ext = int((rain.rain_mm > 150).sum())
if n_ext:
    P(f"- 시간강우 >150mm {n_ext}건 (참고: 관측 극값 확인 필요, 값은 유지)")
    P("  " + " / ".join(f"{r.stn}:{r.tm:%Y-%m-%d %H}시 {r.rain_mm:.1f}mm"
                        for _, r in rain[rain.rain_mm > 150].head(8).iterrows()))

rain.to_parquet(OUT / "강우_AWS시간_long.parquet", index=False)
P(f"- 저장: {OUT/'강우_AWS시간_long.parquet'} ({rain.shape})")

# ---- 지점 메타 + 좌표 결합 ----
coord = pd.read_csv(RAW / "12_위험_강우_AWS_부산" / "KMA_지점좌표_ASOS_AWS.csv")
coord.columns = [c.strip().lower() for c in coord.columns]   # STN,NAME,LAT,LON,ALT_M,TYPE
cov = (rain.groupby("stn")
           .agg(n_hours=("rain_mm", "size"), yr_min=("tm", lambda s: s.min().year),
                yr_max=("tm", lambda s: s.max().year),
                rain_tot=("rain_mm", "sum"))
           .reset_index())
meta = coord.merge(cov, on="stn", how="outer").sort_values("stn")
meta.to_csv(OUT / "강우_AWS_지점.csv", index=False, encoding="utf-8-sig")
P(f"\n## 지점별 커버리지")
P("| 지점 | 명 | 유형 | 좌표 | 시간수 | 연도범위 | 총강우mm |")
P("|--:|---|---|---|--:|---|--:|")
for _, r in meta.iterrows():
    ll = f"{r.lat:.3f},{r.lon:.3f}" if pd.notna(r.get("lat")) else "**좌표없음**"
    yr = f"{int(r.yr_min)}~{int(r.yr_max)}" if pd.notna(r.get("yr_min")) else "데이터없음"
    nh = f"{int(r.n_hours):,}" if pd.notna(r.get("n_hours")) else "0"
    rt = f"{r.rain_tot:,.0f}" if pd.notna(r.get("rain_tot")) else "-"
    P(f"| {int(r.stn)} | {r.get('name','')} | {r.get('type','')} | {ll} | {nh} | {yr} | {rt} |")

miss_coord = meta[meta.lat.isna() & meta.n_hours.notna()]
miss_data = meta[meta.n_hours.isna() & meta.lat.notna()]
if len(miss_coord):
    P(f"\n⚠ 데이터는 있으나 좌표 없는 지점: {miss_coord.stn.tolist()} → stn_inf.php 로 보완 필요")
if len(miss_data):
    P(f"⚠ 좌표는 있으나 시간데이터 없는 지점: {miss_data.stn.tolist()}")

# ---- 침수흔적 사건일과의 커버리지 교차 ----
try:
    import geopandas as gpd
    fl = gpd.read_file(ROOT / "공공데이터" / "가공데이터" / "00_정합_5186" / "침수흔적도.gpkg")
    fl["d"] = pd.to_datetime(fl["FLDN_BGNG_YMD"], format="%Y%m%d", errors="coerce")
    ev_days = fl["d"].dropna().dt.normalize().drop_duplicates().sort_values()
    P(f"\n## 침수흔적 사건일 {len(ev_days)}일 — 지점별 해당일 강우 관측 가용")
    rain["day"] = rain.tm.dt.normalize()
    hit = rain[rain.day.isin(ev_days)].groupby("stn").day.nunique()
    P("| 지점 | 사건일 관측커버 | 비율 |")
    P("|--:|--:|--:|")
    for stn in sorted(rain.stn.unique()):
        h = int(hit.get(stn, 0))
        P(f"| {stn} | {h}/{len(ev_days)} | {h/len(ev_days):.0%} |")
    # 사건일별 최소 1개 지점이라도 관측되는가
    any_cov = rain[rain.day.isin(ev_days)].day.nunique()
    P(f"\n→ 사건일 중 **최소 1개 지점 관측 가용**: {any_cov}/{len(ev_days)} ({any_cov/len(ev_days):.0%})")
except Exception as e:
    P(f"\n(침수흔적 사건일 교차 생략: {e})")

if bad:
    P(f"\n## 파싱 실패 {len(bad)}건")
    for nm, why in bad[:20]:
        P(f"  - {nm}: {why}")

(REP / "S10_강우AWS시간.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'S10_강우AWS시간.md'}", flush=True)
