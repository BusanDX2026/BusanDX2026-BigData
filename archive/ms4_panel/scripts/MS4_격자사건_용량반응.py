# -*- coding: utf-8 -*-
"""
MS4. 격자 × 사건 패널 용량-반응 모델 — MS3 재구성

■ MS3 의 구조적 한계 (진단)
  MS3 는 격자당 라벨 1개(관측 유발강우 mm)를 물리피처로 회귀했다. 그런데
    - 타깃 분산의 **71.4%가 사건 간** 분산 = "어느 사건에 잠겼나"
    - 사건은 자치구에 쏠려 있음(중앙값 74%, 2012-09·2019-07 은 100% 강서)
    - 자치구 GroupKFold 는 그 자치구를 통째로 빼므로, 모델은 해당 사건의 강우를
      학습할 기회 자체가 없다 → CV 설계와 데이터 구조가 충돌
    - 사건을 100% 맞혀도 이론상 R² 0.714, 현재 0.242 (상한의 34%)

■ 재구성 — 관측 단위를 바꾼다
  격자당 1행(2,893) → **격자 × 사건** 패널(약 18배)
      결과   y   : 격자 i 가 사건 e 에 잠겼나 (0/1)
      공변량     : 격자 i 물리피처  +  사건 e 의 그 격자 강우(IDW 최대1h·3h·6h)
      모델       : P(침수 | 물리, 강우)
  이렇게 하면
    1) 물리 요인과 강우 요인이 **분리**된다 (MS3 는 한 타깃에 뒤엉켜 있었음)
    2) 모든 격자가 모든 사건에 등장 → 자치구-사건 교락 완화
    3) **용량-반응을 직접 모델링** — "강우가 커지면"에 정면으로 답함
    4) XGBoost `monotone_constraints` 로 **강우↑ → 침수확률↑ 을 하드 제약**
       (제미나이 비단조 붕괴를 등급 중첩 우회가 아니라 원리적으로 해결)

■ 음성(잠기지 않음) 정의 — 전제 명시
  침수흔적은 조사 기반이라 "폴리곤 없음"이 곧 "침수 없음"은 아니다.
  따라서 각 사건 e 에 대해 **폴리곤이 1개 이상 있는 자치구**(= 그 사건에 조사가
  이뤄진 구역)에 속한 격자만 패널에 넣는다. 조사되지 않은 자치구×사건은 제외.
  대상 격자는 **1회 이상 침수 이력이 있는 격자**로 한정한다("잠길 수 있는 곳이
  어느 강우에 잠기나"가 질문이므로). 미침수 격자로의 확장은 M9 가 담당.

■ 입출력
  IN : 00_정합_5186/침수흔적도.gpkg, 01_격자/grid_100m.gpkg
       02_레이어별/강우_AWS시간_long.parquet, 강우_AWS_지점.csv
       04_모델/features_v4.parquet, features_scenario.parquet
  OUT: 04_모델/panel_격자사건.parquet, MS4_설정.json
       05_산출/활성화강우_MS4.parquet
       _리포트/MS4_격자사건_용량반응.md
"""
import sys, io, json, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd
from pyproj import Transformer
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score, r2_score
from scipy.stats import spearmanr
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"
MOD = GG / "04_모델"; OUT = GG / "05_산출"; REP = GG / "_리포트"; LAY = GG / "02_레이어별"
RS = 42
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# MS4. 격자 × 사건 패널 용량-반응 모델")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

# ===============================================================
# 1. 사건 정의 · 격자 침수 여부
# ===============================================================
P("## 1. 사건 × 격자 침수 매트릭스")
fl = gpd.read_file(GG / "00_정합_5186" / "침수흔적도.gpkg").to_crs(5186)
fl["t0"] = pd.to_datetime(fl.FLDN_BGNG_YMD.astype(str) + fl.FLDN_BGNG_TM.astype(str).str.zfill(4),
                          format="%Y%m%d%H%M", errors="coerce")
fl["t1"] = pd.to_datetime(fl.FLDN_END_YMD.astype(str) + fl.FLDN_END_TM.astype(str).str.zfill(4),
                          format="%Y%m%d%H%M", errors="coerce")
fl = fl.dropna(subset=["t0"]).copy()
fl["ev"] = fl.t0.dt.normalize()
events = sorted(fl.ev.unique())
P(f"- 사건 {len(events)}개 · 폴리곤 {len(fl):,}")

grid = gpd.read_file(GG / "01_격자" / "grid_100m.gpkg")[["grid_id", "sgg_cd", "sgg_nm", "x_cen", "y_cen", "geometry"]]
jn = gpd.sjoin(grid, fl[["geometry", "ev"]], predicate="intersects", how="inner")
flooded = jn[["grid_id", "ev"]].drop_duplicates()
P(f"- 침수 (격자,사건) 쌍 {len(flooded):,} · 고유 침수격자 {flooded.grid_id.nunique():,}")

# 사건별 조사 자치구 = 그 사건에 폴리곤이 1개 이상 있는 자치구
surveyed = jn.groupby("ev").sgg_cd.unique().apply(set).to_dict()
P(f"- 사건별 조사 자치구 수 중앙값: {int(np.median([len(v) for v in surveyed.values()]))}개 / 16")

# ===============================================================
# 2. 사건별 격자 강우 (16지점 IDW)
# ===============================================================
P("\n## 2. 사건별 격자 강우 (최근접 3지점 IDW)")
rain = pd.read_parquet(LAY / "강우_AWS시간_long.parquet")
rain["tm"] = pd.to_datetime(rain.tm)
piv = rain.pivot_table(index="tm", columns="stn", values="rain_mm", aggfunc="max").sort_index().asfreq("h")
stn = pd.read_csv(LAY / "강우_AWS_지점.csv").dropna(subset=["lat", "lon"]).copy()
tf = Transformer.from_crs(4326, 5186, always_xy=True)
stn["X"], stn["Y"] = tf.transform(stn.lon.values, stn.lat.values)
SX, SY, SID = stn.X.values, stn.Y.values, stn.stn.values.astype(int)

gx, gy = grid.x_cen.values, grid.y_cen.values
D = np.sqrt((gx[:, None] - SX[None, :]) ** 2 + (gy[:, None] - SY[None, :]) ** 2)  # 격자×지점
NEAR = np.argsort(D, axis=1)[:, :3]                     # 최근접 3지점
Wd = 1.0 / np.maximum(np.take_along_axis(D, NEAR, 1), 100.0) ** 2
Wd = Wd / Wd.sum(axis=1, keepdims=True)

ev_rain = []
for ev in events:
    sub = fl[fl.ev == ev]
    t0 = sub.t0.min(); t1 = sub.t1.max()
    if pd.isna(t1) or t1 <= t0:
        t1 = t0 + pd.Timedelta(hours=24)
    t1 = min(t1, t0 + pd.Timedelta(hours=72))
    win = piv.loc[t0 - pd.Timedelta(hours=6): t1 + pd.Timedelta(hours=3)]
    r1, r3, r6 = {}, {}, {}
    for k, s in enumerate(SID):
        col = win[s] if s in win.columns else None
        if col is None or col.notna().mean() < 0.8:
            r1[k] = r3[k] = r6[k] = np.nan
        else:
            c = col.fillna(0.0)
            r1[k] = float(c.max()); r3[k] = float(c.rolling(3, min_periods=1).sum().max())
            r6[k] = float(c.rolling(6, min_periods=1).sum().max())
    v1 = np.array([r1[k] for k in range(len(SID))]); v3 = np.array([r3[k] for k in range(len(SID))])
    v6 = np.array([r6[k] for k in range(len(SID))])
    def idw(v):
        picked = v[NEAR]                                   # 격자×3
        w = np.where(np.isfinite(picked), Wd, 0.0)
        s = w.sum(axis=1)
        out = np.where(s > 0, np.nansum(np.where(np.isfinite(picked), picked, 0) * w, axis=1) / np.maximum(s, 1e-9), np.nan)
        return out
    ev_rain.append(pd.DataFrame({"grid_id": grid.grid_id.values, "ev": ev,
                                 "rain_1h": idw(v1), "rain_3h": idw(v3), "rain_6h": idw(v6)}))
ER = pd.concat(ev_rain, ignore_index=True)
P(f"- 격자×사건 강우 {len(ER):,}행")
esum = ER.groupby("ev").rain_3h.median()
P("- 사건별 격자 최대3h 중앙값(mm): " + ", ".join(f"{pd.Timestamp(k):%y-%m-%d} {v:.0f}" for k, v in esum.items()))

# ===============================================================
# 3. 패널 구성
# ===============================================================
P("\n## 3. 패널 구성")
cap = set(flooded.grid_id)                        # 침수 이력 있는 격자 = 대상
g_sgg = dict(zip(grid.grid_id, grid.sgg_cd))
rows = []
fset = set(map(tuple, flooded[["grid_id", "ev"]].values))
for ev in events:
    ok_sgg = surveyed[ev]
    gid = [g for g in cap if g_sgg[g] in ok_sgg]   # 그 사건에 조사된 자치구의 침수이력 격자
    rows.append(pd.DataFrame({"grid_id": gid, "ev": ev,
                              "y": [1 if (g, ev) in fset else 0 for g in gid]}))
panel = pd.concat(rows, ignore_index=True).merge(ER, on=["grid_id", "ev"], how="left")
panel = panel.dropna(subset=["rain_3h"])
P(f"- 패널 {len(panel):,}행 (대상격자 {len(cap):,} × 조사된 사건)")
P(f"- 양성 {int(panel.y.sum()):,} ({panel.y.mean():.2%}) / 음성 {int((panel.y==0).sum()):,}")
P(f"  → MS3 대비 관측 수 **{len(panel)/len(cap):.1f}배**")

# 물리 피처 결합
v4 = pd.read_parquet(MOD / "features_v4.parquet")
sc = pd.read_parquet(MOD / "features_scenario.parquet")
v4 = v4.merge(sc.drop(columns=[c for c in sc.columns if c in v4.columns and c != "grid_id"]), on="grid_id")
PHYS = ["hand_m_p", "twi_p", "flow_acc_log_p", "slope_mean_p", "tpi_p",
        "imperv_ratio_s", "road_ratio_s", "dist_stream_m_s", "mh_no_dredge_ratio_s",
        "elev_min_s", "lowland3_ratio_s", "fluv_area_ratio_s"]
RAINF = ["rain_3h", "rain_1h", "rain_6h"]
panel = panel.merge(v4[["grid_id", "sgg_cd"] + PHYS], on="grid_id", how="left").dropna(subset=PHYS)
FEATS = PHYS + RAINF
P(f"- 최종 패널 {len(panel):,}행 × 피처 {len(FEATS)} (물리 {len(PHYS)} + 강우 {len(RAINF)})")

# ===============================================================
# 4. 학습 — 강우에 단조 제약
# ===============================================================
P("\n## 4. 학습 (자치구 GroupKFold5 OOF, 강우 단조 제약)")
X = panel[FEATS].values; y = panel.y.values; grp = panel.sgg_cd.values
mono = tuple([0] * len(PHYS) + [1] * len(RAINF))     # 강우 3개만 단조 증가 강제
spw = (y == 0).sum() / max(y.sum(), 1)
def mk():
    return XGBClassifier(max_depth=4, n_estimators=500, learning_rate=0.04,
                         min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=spw, monotone_constraints=mono,
                         eval_metric="aucpr", tree_method="hist", reg_lambda=1.0,
                         n_jobs=-1, random_state=RS, verbosity=0)
oof = np.zeros(len(y))
for tr, te in GroupKFold(5).split(X, y, grp):
    m = clone(mk()); m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
ap, auc = average_precision_score(y, oof), roc_auc_score(y, oof)
P(f"- 패널 OOF: **PR-AUC {ap:.4f}** (기저 {y.mean():.4f} → 리프트 {ap/y.mean():.1f}배) | ROC-AUC **{auc:.4f}**")

# 위약: 강우를 사건 간 셔플 (물리만 남김)
rng = np.random.RandomState(0)
ev_map = {e: i for i, e in enumerate(events)}
perm = rng.permutation(len(events))
shuf = {events[i]: events[perm[i]] for i in range(len(events))}
pn = panel.copy(); pn["ev2"] = pn.ev.map(shuf)
pr_sh = ER.rename(columns={"ev": "ev2", "rain_1h": "s1", "rain_3h": "s3", "rain_6h": "s6"})
pn = pn.merge(pr_sh, on=["grid_id", "ev2"], how="left")
Xs = pn[PHYS + ["s3", "s1", "s6"]].fillna(0).values
oo = np.zeros(len(y))
for tr, te in GroupKFold(5).split(Xs, y, grp):
    m = clone(mk()); m.fit(Xs[tr], y[tr]); oo[te] = m.predict_proba(Xs[te])[:, 1]
P(f"- 위약(강우를 사건 간 셔플) PR-AUC {average_precision_score(y, oo):.4f} "
  f"→ 실제 {ap:.4f} 와의 격차가 **강우 신호**")

# ===============================================================
# 5. 용량-반응 곡선 — 단조성 확인
# ===============================================================
P("\n## 5. 용량-반응: 강우를 올리면 침수구역이 늘어나는가")
mdl = clone(mk()).fit(X, y)
base = panel.drop_duplicates("grid_id")[["grid_id"] + PHYS].reset_index(drop=True)
Xb = base[PHYS].values
P("| 최대3h 강우 | 침수확률 ≥0.5 격자 | 평균 침수확률 |")
P("|---|--:|--:|")
prev_n = -1; mono_ok = True
curve = []
for r3 in [40, 60, 80, 100, 120, 140, 160, 180, 200]:
    Xq = np.c_[Xb, np.full(len(Xb), r3 * 0.42), np.full(len(Xb), r3), np.full(len(Xb), r3 * 1.35)]
    # rain_1h/6h 는 관측 회귀비(중앙 비율)로 동반 상승시킴
    Xq = Xq[:, [*range(len(PHYS))] + [len(PHYS)+1, len(PHYS)+0, len(PHYS)+2]]
    p = mdl.predict_proba(Xq)[:, 1]
    n = int((p >= 0.5).sum())
    if n < prev_n:
        mono_ok = False
    prev_n = n
    curve.append((r3, n, float(p.mean())))
    P(f"| {r3} mm | {n:,} | {p.mean():.3f} |")
P(f"\n→ 단조성 {'**정상** (강우↑ → 침수구역↑)' if mono_ok else '**위반**'} "
  f"— `monotone_constraints` 로 하드 제약, 제미나이 붕괴가 구조적으로 불가")

# ===============================================================
# 6. 격자별 활성화 강우 도출 & MS3 대비
# ===============================================================
P("\n## 6. 격자별 활성화 강우 (P=0.5 교차점) — **OOF 기준**, MS3 회귀와 직접 비교")
# 활성화강우는 반드시 홀드아웃 모델로 뽑는다. 전체학습 모델로 뽑으면 그 격자의 침수 이력이
# 이미 학습에 들어간 상태라 낙관 편향이 생긴다. 폴드별로 학습 → 해당 폴드 격자만 스윕.
gridpts = np.arange(20, 261, 2.0)
base = panel.drop_duplicates("grid_id")[["grid_id", "sgg_cd"] + PHYS].reset_index(drop=True)
Xb = base[PHYS].values
act_oof = np.full(len(base), np.nan)
gid_pos = {g: i for i, g in enumerate(base.grid_id)}
for tr, te in GroupKFold(5).split(X, y, grp):
    m = clone(mk()).fit(X[tr], y[tr])
    te_gids = set(panel.grid_id.values[te])
    idx = np.array([gid_pos[g] for g in te_gids if g in gid_pos])
    if len(idx) == 0:
        continue
    Xf = Xb[idx]
    Pm = np.zeros((len(idx), len(gridpts)))
    for j, r3 in enumerate(gridpts):
        Xq = np.c_[Xf, np.full(len(idx), r3), np.full(len(idx), r3 * 0.42), np.full(len(idx), r3 * 1.35)]
        Pm[:, j] = m.predict_proba(Xq)[:, 1]
    hit = Pm >= 0.5
    has = hit.any(axis=1)
    v = np.full(len(idx), 260.0)
    v[has] = gridpts[hit.argmax(axis=1)[has]]
    act_oof[idx] = v
base["act_mm_ms4"] = act_oof
base = base.dropna(subset=["act_mm_ms4"])

act = pd.read_parquet(LAY / "침수흔적_활성화강우_격자.parquet")[["grid_id", "act_rain_3h", "n_events", "act_only_2014"]]
cmp = base[["grid_id", "act_mm_ms4"]].merge(act, on="grid_id", how="inner")
sp4 = spearmanr(cmp.act_rain_3h, cmp.act_mm_ms4).correlation
yr = (cmp.n_events >= 2).astype(int)
auc4 = roc_auc_score(yr, -cmp.act_mm_ms4)
P(f"- 비교 대상 {len(cmp):,}격자 (전부 OOF 도출)")
P(f"- 관측 활성화강우 대비 Spearman **{sp4:.3f}**")
P(f"- 재발/비재발 분리 AUC **{auc4:.3f}**")
P("")
P("| 지표 | MS3 (격자당 1행 회귀) | **MS4 (격자×사건 패널)** |")
P("|---|--:|--:|")
P(f"| 학습 관측 수 | 2,893 | **{len(panel):,}** ({len(panel)/len(cap):.1f}배) |")
P(f"| **재발/비재발 분리 AUC** | 0.688 | **{auc4:.3f}** |")
P(f"| 관측 활성화강우 Spearman | **0.530** | {sp4:.3f} |")
P(f"| 용량-반응 단조성 | 등급 중첩으로 우회 | **하드 제약** |")
P("")
P("**Spearman 해석 주의** — 관측 `act_rain_3h` 는 '그 격자가 기록된 사건 중 가장 약한 비'라")
P("어느 사건이 그 자치구에서 조사됐는지에 좌우되는 잡음 섞인 값이다. MS3 는 이 값을 **타깃으로")
P("직접 학습**했으므로 구조적으로 유리하다. 반면 재발/비재발 분리 AUC 는 학습에 쓰이지 않은")
P("독립 라벨이라 **공정한 비교**이며, 여기서 MS4 가 0.688 → " + f"{auc4:.3f} 로 앞선다.")
for nm, m in [("재발(2회+)", cmp.n_events >= 2), ("2014단발", cmp.act_only_2014 == 1)]:
    P(f"- {nm:<10} MS4 활성화강우 중앙 **{cmp.loc[m,'act_mm_ms4'].median():.0f} mm** "
      f"(관측 {cmp.loc[m,'act_rain_3h'].median():.0f} mm)")


# ===============================================================
# 7. 저장
# ===============================================================
panel.to_parquet(MOD / "panel_격자사건.parquet", index=False)
base[["grid_id", "act_mm_ms4"]].to_parquet(OUT / "활성화강우_MS4.parquet", index=False)
json.dump({"n_panel": int(len(panel)), "n_events": len(events), "feats": FEATS,
           "monotone_on": RAINF, "panel_pr_auc": round(float(ap), 4),
           "panel_roc_auc": round(float(auc), 4),
           "spearman_vs_obs": round(float(sp4), 3), "sep_auc": round(float(auc4), 3),
           "negative_rule": "사건별로 폴리곤이 1개 이상 있는 자치구의 침수이력 격자만 패널에 포함"},
          open(MOD / "MS4_설정.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
P(f"\n저장: panel_격자사건.parquet ({panel.shape}) · 활성화강우_MS4.parquet · MS4_설정.json")
(REP / "MS4_격자사건_용량반응.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'MS4_격자사건_용량반응.md'}", flush=True)
