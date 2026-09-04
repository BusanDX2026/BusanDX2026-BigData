# -*- coding: utf-8 -*-
"""
S11. 침수흔적 폴리곤 ↔ 관측 유발강우 매칭 → 격자 활성화강우 라벨

방법
  479개 침수흔적 폴리곤 각각:
    - 사건창 = [FLDN_BGNG_YMD+TM, FLDN_END_YMD+TM]  (역전 시 +24h, 최대 72h)
    - 선행창 = [사건시작 -48h, 사건시작]
    - 폴리곤 중심 최근접 3개 AWS/ASOS 지점(해당 기간 관측 있는) IDW(1/d²) 시간강우 합성
    - 추출: 총강우 / 최대1h / 최대3h / 최대6h / 최대24h / 선행48h
    - 원인 분류: FLDN_CS_DTL_NM → 내수 / 외수(하천·월류·해일) / 복합 / 미상
  격자 배정(폴리곤∩grid_100m):
    - 셀이 여러 사건에 걸리면 → 사건별 유발강우 중 **최소 최대1h**(= 가장 약한 비에도 잠긴 사건)
      을 활성화강우 라벨로, 함께 그 사건의 날짜·원인·침수심 기록
    - n_events, is_2014, max_depth 병기

입력
  00_정합_5186/침수흔적도.gpkg  (EPSG:5186, BGNG/END YMD·TM, FLDN_CS_DTL_NM, FLDN_DOWA)
  01_격자/grid_100m.gpkg
  02_레이어별/강우_AWS시간_long.parquet, 강우_AWS_지점.csv
출력
  02_레이어별/침수흔적_유발강우_폴리곤.csv   (479행, 진단·검증용)
  02_레이어별/침수흔적_활성화강우_격자.parquet (침수격자 라벨)
  _리포트/S11_유발강우매칭.md
"""
import sys, io, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"
LAY = GG / "02_레이어별"
REP = GG / "_리포트"
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# S11. 침수흔적 ↔ 유발강우 매칭")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

# ---------------------------------------------------------------
# 0. 로드
# ---------------------------------------------------------------
fl = gpd.read_file(GG / "00_정합_5186" / "침수흔적도.gpkg").to_crs(5186)
rain = pd.read_parquet(LAY / "강우_AWS시간_long.parquet")
rain["tm"] = pd.to_datetime(rain["tm"])
stn = pd.read_csv(LAY / "강우_AWS_지점.csv")
stn = stn.dropna(subset=["lat", "lon"]).copy()
tf = Transformer.from_crs(4326, 5186, always_xy=True)
stn["X"], stn["Y"] = tf.transform(stn["lon"].values, stn["lat"].values)
P(f"- 침수흔적 폴리곤 {len(fl):,} · 강우 {len(rain):,}행 / {rain.stn.nunique()}지점 · 좌표지점 {len(stn)}")

# 지점별 시간 인덱스(빠른 조회) — pivot: index=tm, columns=stn
piv = rain.pivot_table(index="tm", columns="stn", values="rain_mm", aggfunc="max").sort_index()
piv = piv.asfreq("h")   # 연속 시간축
P(f"- 강우 피벗 {piv.shape} ({piv.index.min():%Y-%m} ~ {piv.index.max():%Y-%m})")

# ---------------------------------------------------------------
# 1. 원인 분류
# ---------------------------------------------------------------
def cause_cat(s):
    s = str(s)
    ext = any(k in s for k in ["하천", "월류", "범람", "본류", "지류", "해일", "월파", "해안", "만조", "밀물", "해수"])
    inn = any(k in s for k in ["배수", "역류", "내수", "우수", "빗물", "저지대", "용량", "통수"])
    if ext and inn:
        return "복합"
    if ext:
        return "외수"
    if inn:
        return "내수"
    return "미상"
fl["cause"] = fl["FLDN_CS_DTL_NM"].map(cause_cat)
P("- 원인 분류: " + ", ".join(f"{k} {v}" for k, v in fl.cause.value_counts().items()))

# ---------------------------------------------------------------
# 2. 폴리곤별 유발강우 추출
# ---------------------------------------------------------------
def ymd_tm(ymd, tm):
    ymd = str(ymd); tm = str(tm).zfill(4)
    return pd.to_datetime(f"{ymd} {tm[:2]}:{tm[2:]}", format="%Y%m%d %H:%M", errors="coerce")

recs = []
XY = np.c_[stn.X.values, stn.Y.values]
for i, r in fl.iterrows():
    c = r.geometry.centroid
    t0 = ymd_tm(r.FLDN_BGNG_YMD, r.FLDN_BGNG_TM)
    t1 = ymd_tm(r.FLDN_END_YMD, r.FLDN_END_TM)
    if pd.isna(t0):
        recs.append(dict(idx=i, ok=False)); continue
    if pd.isna(t1) or t1 <= t0:
        t1 = t0 + pd.Timedelta(hours=24)
    t1 = min(t1, t0 + pd.Timedelta(hours=72))
    a0 = t0 - pd.Timedelta(hours=48)

    d = np.hypot(XY[:, 0] - c.x, XY[:, 1] - c.y)
    near = np.argsort(d)[:5]           # 후보 5, 데이터 있는 것 중 3 사용
    win = piv.loc[a0:t1 + pd.Timedelta(hours=3)]
    used, ws, series = [], [], []
    for k in near:
        s_id = int(stn.iloc[k].stn)
        if s_id not in win.columns:
            continue
        col = win[s_id]
        if col.empty or col.notna().mean() < 0.8:   # 빈 창(기록범위 밖)·관측률 80% 미만 제외
            continue
        used.append(s_id); ws.append(1.0 / max(d[k], 100.0) ** 2); series.append(col.fillna(0.0))
        if len(used) == 3:
            break
    if not used:
        recs.append(dict(idx=i, ok=False)); continue
    ws = np.array(ws) / sum(ws)
    comb = sum(w * s for w, s in zip(ws, series))     # 합성 시간강우 (a0 ~ t1+3h)

    ev = comb.loc[t0:t1]
    ante = comb.loc[a0:t0 - pd.Timedelta(hours=1)]   # t0 는 사건창 소속 → 중복 제외
    roll = lambda h: comb.rolling(h, min_periods=1).sum()
    recs.append(dict(
        idx=i, ok=True,
        t0=t0, t1=t1, depth=float(pd.to_numeric(r.FLDN_DOWA, errors="coerce")),
        grd=int(r.FLDN_GRD) if pd.notna(r.FLDN_GRD) else np.nan,
        n_stn=len(used), stns="|".join(map(str, used)), near_km=round(float(d[near[0]]) / 1000, 2),
        rain_event=float(ev.sum()),
        rain_1h=float(comb.loc[t0:t1 + pd.Timedelta(hours=3)].max()),
        rain_3h=float(roll(3).loc[t0:t1 + pd.Timedelta(hours=3)].max()),
        rain_6h=float(roll(6).loc[t0:t1 + pd.Timedelta(hours=3)].max()),
        rain_24h=float(roll(24).loc[t0:t1 + pd.Timedelta(hours=3)].max()),
        rain_ante48=float(ante.sum()),
    ))

pr = pd.DataFrame(recs)
ok = pr[pr.ok == True].drop(columns=["ok"]).copy()
fail = pr[pr.ok != True]
P(f"\n- 폴리곤 매칭 성공 {len(ok)}/{len(fl)}  (실패 {len(fail)} = 관측 부족)")
flj = fl.reset_index().rename(columns={"index": "idx"})
okg = flj.merge(ok, on="idx", how="inner")

P("\n## 사건일별 유발강우 (합성 IDW)")
P("| 사건일 | 폴리곤 | 원인우세 | 최대1h(mm) | 최대3h | 총강우 | 선행48h | 침수심중앙 |")
P("|---|--:|---|--:|--:|--:|--:|--:|")
for day, g in okg.groupby(okg.t0.dt.date):
    P(f"| {day} | {len(g)} | {g.cause.mode().iat[0]} | {g.rain_1h.median():.0f} | {g.rain_3h.median():.0f} "
      f"| {g.rain_event.median():.0f} | {g.rain_ante48.median():.0f} | {g.depth.median():.2f} |")

okg.drop(columns=["geometry"]).to_csv(LAY / "침수흔적_유발강우_폴리곤.csv", index=False, encoding="utf-8-sig")
P(f"\n- 저장: {LAY/'침수흔적_유발강우_폴리곤.csv'} ({len(okg)}행)")

# ---------------------------------------------------------------
# 3. 격자 배정 (폴리곤 ∩ grid_100m)
# ---------------------------------------------------------------
grid = gpd.read_file(GG / "01_격자" / "grid_100m.gpkg")[["grid_id", "geometry"]]
jn = gpd.sjoin(grid,
               okg.set_geometry("geometry")[["geometry", "t0", "cause", "depth", "rain_event",
                                             "rain_1h", "rain_3h", "rain_6h", "rain_24h", "rain_ante48"]],
               predicate="intersects", how="inner")
P(f"\n- 폴리곤∩격자 교차 {len(jn):,}쌍, 고유 침수격자 {jn.grid_id.nunique():,}")

jn["ev_date"] = jn.t0.dt.normalize()      # 사건 = 날짜 (같은 폭우의 14시/15시/17시 폴리곤을 1건으로)
jn["is_2014"] = (jn.t0.dt.year == 2014).astype(int)
# 셀별: 가장 '약한 비'에 잠긴 사건 = rain_1h 최소인 행을 활성화 라벨로
jn = jn.sort_values(["grid_id", "rain_1h"])
cell = jn.groupby("grid_id").agg(
    n_events=("ev_date", "nunique"),
    act_date=("t0", "first"), act_cause=("cause", "first"),
    act_rain_1h=("rain_1h", "first"), act_rain_3h=("rain_3h", "first"),
    act_rain_event=("rain_event", "first"), act_rain_ante48=("rain_ante48", "first"),
    max_rain_1h=("rain_1h", "max"), max_depth=("depth", "max"),
    is_2014_any=("is_2014", "max"),
).reset_index()
cell["act_only_2014"] = ((cell.n_events == 1) & (cell.is_2014_any == 1)).astype(int)

cell.to_parquet(LAY / "침수흔적_활성화강우_격자.parquet", index=False)
P(f"- 저장: {LAY/'침수흔적_활성화강우_격자.parquet'} ({cell.shape})")

P("\n## 활성화강우(최대1h) 분포 — 재발 vs 2014단발")
for nm, m in [("재발(2회+)", cell.n_events >= 2), ("2014단발", cell.act_only_2014 == 1),
              ("기타단발", (cell.n_events == 1) & (cell.act_only_2014 == 0))]:
    s = cell.loc[m, "act_rain_1h"]
    P(f"  {nm:<10} n={m.sum():4d}  최대1h  p25/p50/p75 = {s.quantile(.25):.0f} / {s.quantile(.5):.0f} / {s.quantile(.75):.0f} mm")
P("\n  → 재발 격자가 더 낮은 강우에 활성화되면 = 관측이 물리 가설을 지지")

if len(fail):
    fday = flj.loc[fail.idx, "FLDN_BGNG_YMD"].value_counts()
    P(f"\n## 매칭 실패 {len(fail)}건 — 사건일별: {fday.to_dict()}")

(REP / "S11_유발강우매칭.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'S11_유발강우매칭.md'}", flush=True)
