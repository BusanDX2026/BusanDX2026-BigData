# -*- coding: utf-8 -*-
"""
MS3. 관측 유발강우 회귀 → 격자별 활성화 강우(mm/hr) → 강우 시나리오 등급 재구성

MS2와의 차이
  MS2: 활성화 축 = HAND+TWI+s_phys 물리 결합. 재발/극한 분리 AUC 0.55 (느슨, 광역 밴드).
  MS3: 활성화 축 = 침수흔적 격자의 **관측 유발강우(최대1h mm)** 를 물리피처로 회귀 학습.
       → 등급 경계를 백분위가 아니라 **측정된 mm/hr 임계**로 정의.
       관측: 재발격자 최대1h 중앙 47mm vs 2014단발 85mm (S11).

입력
  02_레이어별/침수흔적_활성화강우_격자.parquet  (2,893 침수격자: act_rain_3h 등)
  04_모델/features_v4.parquet, features_scenario.parquet, hazard_score.parquet
출력
  05_산출/시나리오_활성화등급_v2.parquet, 시나리오_행정동_우선순위_v2.csv
  04_모델/MS3_설정.json, _리포트/MS3_활성화강우_관측회귀.md
"""
import sys, io, json, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"
MOD = GG / "04_모델"; OUT = GG / "05_산출"; REP = GG / "_리포트"
LAY = GG / "02_레이어별"
RS = 42
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# MS3. 관측 유발강우 회귀 → 활성화 강우 등급")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

# ---------------------------------------------------------------
# 0. 결합
# ---------------------------------------------------------------
df = pd.read_parquet(MOD / "features_v4.parquet")
sc = pd.read_parquet(MOD / "features_scenario.parquet")
hz = pd.read_parquet(MOD / "hazard_score.parquet")[["grid_id", "hazard_oof"]]
act = pd.read_parquet(LAY / "침수흔적_활성화강우_격자.parquet")
df = (df.merge(sc.drop(columns=[c for c in sc.columns if c in df.columns and c != "grid_id"]), on="grid_id")
        .merge(hz, on="grid_id").merge(act, on="grid_id", how="left"))
df["m9_pct"] = df.hazard_oof.rank(pct=True)
P(f"격자 {len(df):,} · 관측 활성화강우 라벨 있는 침수격자 {df.act_rain_3h.notna().sum():,}")

# ---------------------------------------------------------------
# 1. 관측 유발강우(최대3h) 회귀 — 침수격자, 자치구 GroupKFold5
#    최대3h 선택 이유: 물리피처 회귀력이 1h(R²0.07)보다 3h(R²0.15)가 높다.
#    도시배수는 분단위 첨두는 버퍼링하고 수시간 지속부하에서 무너지므로 3h가 물리적으로도 타당.
# ---------------------------------------------------------------
P("\n## 1. 유발강우(최대3h mm) 회귀 [XGBRegressor, 자치구 GroupKFold5 OOF]")
FEATS = ["hand_m_p", "twi_p", "flow_acc_log_p", "slope_mean_p", "tpi_p",
         "imperv_ratio_s", "road_ratio_s", "dist_stream_m_s", "mh_no_dredge_ratio_s",
         "elev_min_s", "lowland3_ratio_s", "fluv_area_ratio_s"]
tr = df[df.act_rain_3h.notna()].copy()
X, y, g = tr[FEATS].values, tr.act_rain_3h.values, tr.sgg_cd.values
gkf = list(GroupKFold(n_splits=5).split(X, y, g))
reg = XGBRegressor(max_depth=3, n_estimators=400, learning_rate=0.03, min_child_weight=20,
                   subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                   tree_method="hist", n_jobs=-1, random_state=RS, objective="reg:squarederror")
oof = np.zeros(len(y))
for a, b in gkf:
    m = clone(reg); m.fit(X[a], y[a]); oof[b] = m.predict(X[b])
mae, r2 = mean_absolute_error(y, oof), r2_score(y, oof)
rho = spearmanr(y, oof).correlation
base_mae = mean_absolute_error(y, np.full_like(y, np.median(y)))
P(f"- OOF  MAE {mae:.1f} mm  (기저=중앙값예측 MAE {base_mae:.1f}) | R² {r2:.3f} | Spearman {rho:.3f}")
# 재발/2014단발 예측 분리 + 관측 앵커 산출
tr["_oof"] = oof
ANCH = {}
for nm, m in [("재발(2회+)", tr.n_events >= 2), ("2014단발", tr.act_only_2014 == 1)]:
    ANCH[nm] = (tr.loc[m, "act_rain_3h"].quantile(.25), tr.loc[m, "act_rain_3h"].median(),
                tr.loc[m, "act_rain_3h"].quantile(.75))
    P(f"  {nm:<10} 관측 최대3h p25/50/75 = {ANCH[nm][0]:.0f}/{ANCH[nm][1]:.0f}/{ANCH[nm][2]:.0f} → 예측중앙 {tr.loc[m,'_oof'].median():.0f} mm")
auc_sep = None
try:
    from sklearn.metrics import roc_auc_score
    mm = tr.n_events >= 2
    y_rec = mm.astype(int).values
    auc_sep = roc_auc_score(y_rec, -oof)   # 예측강우 낮을수록 재발
    P(f"- 예측 유발강우로 재발/비재발 분리 AUC {auc_sep:.3f}  (MS2 물리결합 0.55 대비)")
except Exception as e:
    P(f"  (분리 AUC 생략: {e})")

# 위약 대조 — 라벨 자치구 내부 셔플
rng = np.random.RandomState(0)
y_sh = tr.groupby("sgg_cd")["act_rain_3h"].transform(lambda s: rng.permutation(s.values)).values
oo = np.zeros(len(y))
for a, b in gkf:
    m = clone(reg); m.fit(X[a], y_sh[a]); oo[b] = m.predict(X[b])
P(f"- 위약(자치구내 셔플) OOF R² {r2_score(y_sh, oo):+.3f}  → 실제 R² {r2:.3f} 와의 격차가 물리신호")

# ---------------------------------------------------------------
# 2. 전 격자 활성화 강우 예측 & mm/hr 임계 등급
# ---------------------------------------------------------------
P("\n## 2. 전 격자 활성화 강우(최대3h mm) 예측 → 임계 등급")
reg_full = clone(reg).fit(X, y)
df["act_mm_pred"] = reg_full.predict(df[FEATS].values)
# 침수격자는 관측값을 우선 사용 (예측보다 정확)
df["act_mm"] = df["act_rain_3h"].where(df.act_rain_3h.notna(), df.act_mm_pred)
P(f"- 예측 활성화강우(3h) 분포(전격자) p10/50/90 = {df.act_mm_pred.quantile([.1,.5,.9]).round(0).tolist()} mm/3h")

# universe (MS2와 동일 규칙)
fr = pd.read_parquet(LAY / "홍수위험_grid.parquet")[["grid_id", "pluv_area_ratio"]]
df = df.merge(fr, on="grid_id", how="left"); df["pluv_area_ratio"] = df.pluv_area_ratio.fillna(0)
df["hazdist_flood_active"] = df["hazdist_flood_active"].fillna(0).astype(int)
universe = ((df.m9_pct >= 0.70) | (df.trace_flag == 1) | (df.fluv_area_ratio > 0.10) |
            (df.pluv_area_ratio > 0) | (df.hazdist_flood_active == 1))
df["in_universe"] = universe.astype(int)

# 최대3h(mm) 임계 — 관측 앵커(재발 vs 2014단발 중앙값)와 KMA 호우특보(3h) 기준 결합
#   호우주의보 3h 60mm / 호우경보 3h 90mm 가 행정 기준선.
TH1 = round(float(np.mean([ANCH["재발(2회+)"][2], 90])), 0)     # 재발 p75 와 호우경보(90) 사이
TH2 = round(float(np.mean([ANCH["2014단발"][0], ANCH["2014단발"][1]])), 0)  # 2014단발 p25~중앙
assert TH1 < TH2, f"임계 역전: TH1={TH1} >= TH2={TH2} → T2 밴드가 사라짐 (코드리뷰 #9)"
P(f"- 임계(최대3h): T1≤{TH1:.0f}mm / T2≤{TH2:.0f}mm / T3>{TH2:.0f}mm")
def tier(r):
    if not r.in_universe:
        return 0
    if r.act_mm <= TH1:
        return 1
    if r.act_mm <= TH2:
        return 2
    return 3
df["tier"] = df.apply(tier, axis=1)
# 200년 하천범람 우세 격자는 universe(fluv>0.10)에 이미 포함 → tier>=1 보장 (코드리뷰 #6)
assert (df.loc[df.fluv_area_ratio > 0.30, "tier"] > 0).all(), "200년 하천범람 격자가 대응구역 밖에 있음"

SCEN = {1: ("T1 상시취약", f"최대3h ~{TH1:.0f}mm 이하 (호우경보 문턱)"),
        2: ("T2 집중호우취약", f"최대3h {TH1:.0f}~{TH2:.0f}mm"),
        3: ("T3 극한호우취약", f"최대3h {TH2:.0f}mm 초과 (2014.8.25급)")}

# universe 내 등급 근거 구분 — 관측(측정) vs 회귀예측
n_obs = int(((df.tier > 0) & df.act_rain_3h.notna()).sum())
n_pred = int(((df.tier > 0) & df.act_rain_3h.isna()).sum())
P(f"- 등급 근거: 관측 유발강우 {n_obs:,}격자(정밀) / 회귀예측 {n_pred:,}격자 "
  f"(예측 R² {r2:.3f}·Spearman {rho:.2f} → 광역 추정)")
for t, (nm, rr) in SCEN.items():
    n = int((df.tier == t).sum())
    P(f"  {nm:<14} {rr:<26} {n:>6,}격자 (누적 {int(df.tier.between(1,t).sum()):,})")
P(f"  T0 방어        {'':<26} {int((df.tier==0).sum()):>6,}격자")

# ---------------------------------------------------------------
# 3. 검증 A/B — 등급 분포 & 누적 포착(단조)
# ---------------------------------------------------------------
P("\n## 3. 검증 A: 등급 층화 — **OOF 예측 기준**(비순환)")
# 코드리뷰 #7: 구버전은 침수격자의 tier 를 '관측 유발강우'에서 직접 만든 뒤 그 분포를
#   검증이라 불렀다. 라벨→등급이 결정적 함수라 순환이었다. 여기서는 침수격자 등급을
#   홀드아웃(OOF) 예측으로 다시 매겨 모델이 스스로 재발형/2014단발을 가르는지 본다.
tr_v = tr.copy()
tr_v["tier_oof"] = np.where(tr_v._oof <= TH1, 1, np.where(tr_v._oof <= TH2, 2, 3))
tr_v["tier_obs"] = np.where(tr_v.act_rain_3h <= TH1, 1, np.where(tr_v.act_rain_3h <= TH2, 2, 3))
tr_v["grp"] = np.where(tr_v.n_events >= 2, "재발(2회+)",
               np.where(tr_v.act_only_2014 == 1, "2014단발", "기타단발"))
for col, tag in [("tier_oof", "OOF 예측 기준 — 비순환(검증용)"), ("tier_obs", "관측값 기준 — 순환(참고용)")]:
    ct = pd.crosstab(tr_v.grp, tr_v[col], normalize="index")
    P(f"\n**{tag}**")
    P("| 그룹 | T1 | T2 | T3 |")
    P("|---|--:|--:|--:|")
    for gname in ["재발(2회+)", "기타단발", "2014단발"]:
        if gname in ct.index:
            rr = ct.loc[gname]
            P(f"| {gname} | {rr.get(1,0):.0%} | {rr.get(2,0):.0%} | {rr.get(3,0):.0%} |")
P("\n→ OOF 기준에서도 재발형이 T1 쪽, 2014단발이 T3 쪽으로 몰리면 모델이 강우축을 학습한 것.")
P("  전 격자 등급은 침수이력 격자엔 관측값, 나머지엔 회귀예측을 쓴다(근거를 구분 기록).")

P("\n## 4. 검증 B: 시나리오 누적 포착률 (단조 구조 보장)")
y2 = df.trace_flag.values; tot = y2.sum()
is14 = (df.trace_flag == 1) & (df.act_only_2014 == 1)
rec = (df.trace_flag == 1) & (df.n_events >= 2)
totpop = df["pop"].sum()
P("| 시나리오 | 시간우량 | 대응격자 | 면적 | 전체포착 | 재발형 | 2014급 | 정밀도 | 인구% |")
P("|---|---|--:|--:|--:|--:|--:|--:|--:|")
prev = 0
for t, (nm, rr) in SCEN.items():
    m = df.tier.between(1, t).values
    cap = y2[m].sum() / tot
    assert cap >= prev - 1e-9
    prev = cap
    P(f"| {nm} | {rr.split('(')[0].strip()} | {m.sum():,} | {m.sum()*.01:.0f}km² | **{cap:.1%}** | "
      f"{rec.values[m].sum()/rec.sum():.1%} | {is14.values[m].sum()/is14.sum():.1%} | {y2[m].mean():.1%} | "
      f"{df.loc[m,'pop'].sum()/totpop:.0%} |")

# ---------------------------------------------------------------
# 5. 검증 C — 공간 판별력(M9 대비) & 자치구 내부
# ---------------------------------------------------------------
P("\n## 5. 검증 C: 등급 내부 공간순위(M9)로 좁혀볼 때 포착 · 자치구내 상관")
# 등급 우선(T1>T2>T3>T0) + 등급 내부는 M9 위험도로 정렬
df["tier_score"] = (4 - df.tier.replace(0, 4)) + df.m9_pct
df["scen_score"] = np.where(df.in_universe == 1, df.tier_score, -1.0)
def topn(s, p):
    k = max(1, int(len(s) * p))
    j = s + np.random.RandomState(RS).rand(len(s)) * max(np.ptp(s), 1e-9) * 1e-9
    return y2[np.argpartition(-j, k - 1)[:k]].sum() / tot
P("  (등급 순 → 등급 내부 M9. 시나리오는 '언제', M9는 '등급 안에서 어디')")
for p in [0.05, 0.10, 0.20, 0.25]:
    P(f"  Top{p:.0%}  시나리오점수 {topn(df.scen_score.values,p):.1%}  |  M9 단독 {topn(df.hazard_oof.values,p):.1%}")
tg = pd.read_parquet(GG / "03_마스터" / "target_grid.parquet")[["grid_id", "trace_area_ratio"]]
dd = df.merge(tg, on="grid_id", how="left"); dd["trace_area_ratio"] = dd.trace_area_ratio.fillna(0)
rr = [(s, spearmanr(x.tier_score, x.trace_area_ratio).correlation)
      for s, x in dd.groupby("sgg_nm") if (x.trace_area_ratio > 0).sum() >= 15]
rr = pd.DataFrame(rr, columns=["sgg", "rho"]).sort_values("rho")
P(f"  자치구내 Spearman(tier_score, 침수면적비) 중앙값 {rr.rho.median():+.3f} "
  f"(하위 {rr.iloc[0].sgg} {rr.iloc[0].rho:+.2f} / 상위 {rr.iloc[-1].sgg} {rr.iloc[-1].rho:+.2f})")

# ---------------------------------------------------------------
# 6. 행정동 시나리오 우선순위
# ---------------------------------------------------------------
P("\n## 6. 행정동별 시나리오 우선순위 (T1 노출인구 상위 15)")
rows = []
for (sgg, adm, nm), gg2 in df.groupby(["sgg_nm", "adm_cd", "adm_nm"]):
    rows.append(dict(자치구=sgg, 행정동=nm, 총격자=len(gg2), 인구=int(gg2["pop"].sum()),
                     T1=int((gg2.tier == 1).sum()), T2=int((gg2.tier == 2).sum()), T3=int((gg2.tier == 3).sum()),
                     침수흔적=int(gg2.trace_flag.sum()),
                     T1_pop=int(gg2.loc[gg2.tier == 1, "pop"].sum()),
                     T2누적_pop=int(gg2.loc[gg2.tier.between(1, 2), "pop"].sum()),
                     활성화강우_중앙=round(float(gg2.loc[gg2.in_universe == 1, "act_mm"].median()), 0)
                        if (gg2.in_universe == 1).any() else np.nan))
dong = pd.DataFrame(rows).sort_values("T1_pop", ascending=False).reset_index(drop=True)
P("| 순위 | 자치구 | 행정동 | T1격자 | T1노출인구 | 활성화강우중앙(mm/hr) | 침수흔적 |")
P("|--:|---|---|--:|--:|--:|--:|")
for i, r in dong.head(15).iterrows():
    P(f"| {i+1} | {r.자치구} | {r.행정동} | {r.T1} | {r.T1_pop:,} | {r.활성화강우_중앙:.0f} | {r.침수흔적} |")

# ---------------------------------------------------------------
# 7. 저장
# ---------------------------------------------------------------
keep = ["grid_id", "sgg_nm", "adm_nm", "x_cen", "y_cen", "pop", "trace_flag", "n_events",
        "act_rain_3h", "act_mm_pred", "act_mm", "hand_m", "m9_pct", "tier_score", "in_universe", "tier"]
df[keep].to_parquet(OUT / "시나리오_활성화등급_v2.parquet", index=False)
dong.to_csv(OUT / "시나리오_행정동_우선순위_v2.csv", index=False, encoding="utf-8-sig")
P(f"\n저장: {OUT/'시나리오_활성화등급_v2.parquet'} ({df[keep].shape}) / 행정동 {len(dong)}")
json.dump({"reg_feats": FEATS, "reg_oof": {"mae": round(mae, 2), "r2": round(r2, 3),
           "spearman": round(rho, 3), "sep_auc": round(auc_sep, 3) if auc_sep else None},
           "mm_thresholds": {"T1_max": TH1, "T2_max": TH2},
           "obs_anchor_3h": {"recur": [round(float(x),1) for x in ANCH["재발(2회+)"]], "y2014": [round(float(x),1) for x in ANCH["2014단발"]]},
           "tier_score_formula": "0.70*m9_pct + 0.30*(1 - rank(act_mm))"},
          open(MOD / "MS3_설정.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
(REP / "MS3_활성화강우_관측회귀.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'MS3_활성화강우_관측회귀.md'}", flush=True)
