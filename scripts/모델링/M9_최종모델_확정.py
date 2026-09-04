# -*- coding: utf-8 -*-
"""
M9. 최종 모델 확정 (16피처) — 성능표 · 검증 · hazard_score 갱신

피처 구성 (물리 HAZARD 만)
  지형·수문 9 : elev_min, slope_mean, tpi, lowland3_ratio, fluv_area_ratio,
                rain_annmax_mm, flow_acc_log, twi, dist_stream_m
  토지피복 6  : imperv, agri, paddy, forest, water, road          (M6, +13.4%)
  하수도   1  : mh_no_dredge_ratio  준설 미실시 비율               (M8, +6.8%)

  ※ 하수맨홀 나머지 4종(불량률·합류식·노후도·우수비)은 위약 대조에서 '행정동 지문'으로
    판명되어 폐기 (이슈 #16). SHAP 기여율 19.8% 였으나 단독 성능 −0.3%.
  ※ 인구·건물·펌프장은 여전히 제외 — 보고편향·역인과 (이슈 #9, M2 실험2)

- 입력: 04_모델/features_v4.parquet, M3_최종설정.json
- 출력: 04_모델/hazard_score.parquet (갱신), M9_최종설정.json, M9_shap.csv
        _리포트/M9_최종모델.md
"""
import sys, io, json, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from scipy.stats import spearmanr
from xgboost import XGBClassifier
import shap

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"
MOD = GG / "04_모델"; REP = GG / "_리포트"
RS = 42
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# M9. 최종 모델 확정")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

df = pd.read_parquet(MOD / "features_v4.parquet")
cfg = json.load(open(MOD / "M3_최종설정.json", encoding="utf-8"))
LC = ["imperv_ratio", "agri_ratio", "paddy_ratio", "forest_ratio", "water_ratio", "road_ratio"]
FEATS = cfg["features"] + [c + "_s" for c in LC] + ["mh_no_dredge_ratio_s"]
X = df[FEATS].values; y = df.trace_flag.values; groups = df.sgg_cd.values
gkf = list(GroupKFold(n_splits=5).split(X, y, groups))
P(f"격자 {len(df):,} · 피처 **{len(FEATS)}** · 양성 {y.sum():,} ({y.mean():.2%})")
P(f"피처: {FEATS}\n")

def mk(spw=None):
    return XGBClassifier(**cfg["params"], scale_pos_weight=(spw or cfg["scale_pos_weight"]),
                         eval_metric="aucpr", tree_method="hist", reg_lambda=1.0,
                         n_jobs=-1, random_state=RS, verbosity=0)

def oof_of(X_, y_, splits, spw=None):
    o = np.zeros(len(y_))
    for tr, te in splits:
        m = clone(mk(spw)); m.fit(X_[tr], y_[tr]); o[te] = m.predict_proba(X_[te])[:, 1]
    return o

oof = oof_of(X, y, gkf)
tie = np.random.RandomState(RS).rand(len(y))
s = oof + tie * max(np.ptp(oof), 1e-9) * 1e-9
order = np.argsort(-s); N, POS = len(y), int(y.sum())

P("## 1. 성능 (자치구 GroupKFold5 OOF)")
P(f"- PR-AUC **{average_precision_score(y, oof):.4f}** (양성률 {y.mean():.4f} 대비 **{average_precision_score(y,oof)/y.mean():.1f}배**)")
P(f"- ROC-AUC {roc_auc_score(y, oof):.4f} | Brier {brier_score_loss(y, np.clip(oof,0,1)):.4f}")
P("\n| 상위 | 격자 | 재현율(포착) | 정밀도 | 놓친 침수 |")
P("|---|--:|--:|--:|--:|")
for pct in [0.05, 0.10, 0.20, 0.30]:
    k = int(N * pct); tp = int(y[order[:k]].sum())
    P(f"| {pct:.0%} | {k:,} | **{tp/POS:.1%}** | {tp/k:.1%} | {POS-tp:,} |")

P("\n## 2. 개선 경로 (누적)")
# 2026-09-04 코드리뷰: 기존엔 과거 PR-AUC가 하드코딩돼 있어 상류(수문 라우팅)를 고친 뒤에도
#   낡은 수치를 출력했다. 동일 튜닝 파라미터로 누적 절제를 매 실행 시 직접 계산한다.
PHYS6 = ["elev_min_s", "slope_mean_s", "tpi_s", "lowland3_ratio_s", "fluv_area_ratio_s", "rain_annmax_mm_s"]
HYD3 = ["flow_acc_log_s", "twi_s", "dist_stream_m_s"]
LC6 = ["imperv_ratio_s", "agri_ratio_s", "paddy_ratio_s", "forest_ratio_s", "water_ratio_s", "road_ratio_s"]
MH1 = ["mh_no_dredge_ratio_s"]
STAGES = [("물리 6 (표고·경사·TPI·저지대·하천범람·강우)", PHYS6),
          ("＋수문 3 (흐름누적·TWI·수계거리)", PHYS6 + HYD3),
          ("＋토지피복 6 (불투수면적 등)", PHYS6 + HYD3 + LC6),
          ("＋준설미실시 1 (하수도)", PHYS6 + HYD3 + LC6 + MH1)]
P("| 단계 | 피처수 | PR-AUC | Top10% 포착 |")
P("|---|--:|--:|--:|")
prev = None
for nm, feats in STAGES:
    o_s = oof_of(df[feats].values, y, gkf)
    ap_s = average_precision_score(y, o_s)
    j = o_s + np.random.RandomState(RS).rand(len(y)) * max(np.ptp(o_s), 1e-9) * 1e-9
    t10 = y[np.argpartition(-j, int(N * .10) - 1)[:int(N * .10)]].sum() / POS
    d = f" ({ap_s/prev-1:+.1%})" if prev else ""
    P(f"| {nm} | {len(feats)} | {ap_s:.4f}{d} | {t10:.1%} |")
    prev = ap_s
P(f"\n- 기준선 대비: MCDA 동일가중 0.0480 → **{average_precision_score(y,oof)/0.0480:.1f}배**, "
  f"행정 재해위험지구 0.0377 → **{average_precision_score(y,oof)/0.0377:.1f}배**")

P("\n## 3. [V1] 2014년 사건 일반화 (독립 검증)")
is14 = ((df.trace_flag == 1) & (df.trace_last_year == 2014)).values
y_no14 = np.where(is14, 0, y)
spw2 = (y_no14 == 0).sum() / max(y_no14.sum(), 1)
o14 = oof_of(X, y_no14, list(GroupKFold(n_splits=5).split(X, y_no14, groups)), spw2)
mask = is14 | (y == 0)
ap14 = average_precision_score(y[mask], o14[mask]); base14 = y[mask].mean()
k10 = int(mask.sum() * .10)
s14 = o14[mask] + tie[mask] * max(np.ptp(o14[mask]), 1e-9) * 1e-9
t14 = y[mask][np.argpartition(-s14, k10 - 1)[:k10]].sum() / max(y[mask].sum(), 1)
P(f"- 2014 침수 {int(is14.sum()):,}건을 학습에서 완전 배제 후 예측")
P(f"- PR-AUC {ap14:.4f} (양성률 {base14:.4f}) → **리프트 {ap14/base14:.1f}배** | Top10% {t14:.1%}")
P(f"- 이전(9피처) 리프트 2.6배 → **{ap14/base14:.1f}배**")

P("\n## 4. [V2] 자치구 내 상대순위")
rows = []
for gu in df.sgg_nm.unique():
    m = (df.sgg_nm == gu).values
    if y[m].sum() < 10: continue
    rows.append((gu, spearmanr(oof[m], y[m]).statistic))
gr = pd.DataFrame(rows, columns=["자치구", "rho"]).sort_values("rho", ascending=False)
P(f"- 16개 자치구 전부 ρ>0: {(gr.rho>0).all()} | 중앙값 **{gr.rho.median():+.3f}**")
P(f"- 최고 {gr.iloc[0].자치구} {gr.iloc[0].rho:+.3f} / 최저 {gr.iloc[-1].자치구} {gr.iloc[-1].rho:+.3f}")

P("\n## 5. SHAP")
final = mk().fit(X, y)
samp = df.sample(n=min(20000, len(df)), random_state=RS)
sv = shap.TreeExplainer(final).shap_values(samp[FEATS].values)
imp = pd.DataFrame({"feature": FEATS, "shap": np.abs(sv).mean(0)})
imp["dir"] = [np.corrcoef(samp[f].values, sv[:, i])[0, 1] for i, f in enumerate(FEATS)]
imp["share"] = imp.shap / imp.shap.sum()
imp = imp.sort_values("shap", ascending=False)
imp.to_csv(MOD / "M9_shap.csv", index=False, encoding="utf-8-sig")
P("| 피처 | 기여율 | 방향 |")
P("|---|--:|:-:|")
for _, r in imp.iterrows():
    P(f"| {r.feature} | {r.share:.1%} | {'↑' if r['dir']>0 else '**↓**'} ({r['dir']:+.2f}) |")

df["hazard_raw"] = final.predict_proba(X)[:, 1]
df["hazard_oof"] = oof
df["hazard_pct"] = df.hazard_raw.rank(pct=True)
df[["grid_id", "sgg_cd", "sgg_nm", "adm_cd", "adm_nm", "hazard_raw", "hazard_oof", "hazard_pct"]] \
    .to_parquet(MOD / "hazard_score.parquet", index=False)
json.dump({"features": FEATS, "params": cfg["params"], "scale_pos_weight": cfg["scale_pos_weight"],
           "pr_auc": float(average_precision_score(y, oof)), "n_features": len(FEATS)},
          open(MOD / "M9_최종설정.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
P(f"\n- 산출: hazard_score.parquet(갱신), M9_최종설정.json, M9_shap.csv")

(REP / "M9_최종모델.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'M9_최종모델.md'}", flush=True)
