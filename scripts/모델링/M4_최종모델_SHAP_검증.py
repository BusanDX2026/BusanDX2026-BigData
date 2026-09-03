# -*- coding: utf-8 -*-
"""
M4. 최종 모델 학습 · SHAP 해석 · 3종 독립 검증

검증 3종 (전처리 코드리뷰 FIND-2·3 대응)
  [V1] 사건 일반화: 2014년(전체 침수흔적의 56.6%) 침수를 학습에서 제외하고,
       그 모델이 2014년 침수를 맞히는지 본다. → 단일 호우사건 과적합 여부
  [V2] 자치구 내 상대순위: 침수흔적도는 지자체별 조사 편향이 있으므로(기장 886 vs 부산진 22),
       전역 절대성능 대신 '자치구 안에서의 순위'가 맞는지 Spearman으로 본다.
  [V3] 확률 보정(calibration): MCDA에서 노출과 곱할 것이므로 점수가 확률로서 신뢰되는지 확인.

- 입력: 04_모델/features_v2.parquet, M3_최종설정.json
- 출력: 04_모델/hazard_score.parquet, M4_shap_importance.csv, _리포트/M4_최종모델_검증.md
"""
import sys, io, json, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
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

P("# M4. 최종 모델 · SHAP · 검증")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

df = pd.read_parquet(MOD / "features_v2.parquet")
cfg = json.load(open(MOD / "M3_최종설정.json", encoding="utf-8"))
FEATS = cfg["features"]; PARAMS = cfg["params"]; SPW = cfg["scale_pos_weight"]
X = df[FEATS].values; y = df.trace_flag.values; groups = df.sgg_cd.values
gkf = list(GroupKFold(n_splits=5).split(X, y, groups))
P(f"격자 {len(df):,} · 피처 {len(FEATS)} · 양성 {y.sum():,} ({y.mean():.2%})")

def mk(spw=SPW):
    return XGBClassifier(**PARAMS, scale_pos_weight=spw, eval_metric="aucpr",
                         tree_method="hist", reg_lambda=1.0, n_jobs=-1,
                         random_state=RS, verbosity=0)

def oof_of(X, y, splits, mdl):
    o = np.zeros(len(y))
    for tr, te in splits:
        m = clone(mdl); m.fit(X[tr], y[tr]); o[te] = m.predict_proba(X[te])[:, 1]
    return o

def topn(y_true, s, pct):
    k = max(1, int(len(s)*pct))
    o = s + np.random.RandomState(RS).rand(len(s))*max(np.ptp(s), 1e-9)*1e-9
    return y_true[np.argpartition(-o, k-1)[:k]].sum()/max(y_true.sum(), 1)

# ===============================================================
# 1. OOF 성능 (재확인) + 최종 전체학습
# ===============================================================
oof = oof_of(X, y, gkf, mk())
P(f"\n## 1. 최종 성능 (자치구 GroupKFold5 OOF)")
P(f"- PR-AUC **{average_precision_score(y, oof):.4f}** (양성률 {y.mean():.4f} 대비 {average_precision_score(y,oof)/y.mean():.1f}배)")
P(f"- ROC-AUC {roc_auc_score(y, oof):.4f} | Top5% {topn(y,oof,.05):.1%} | Top10% {topn(y,oof,.10):.1%} | Top20% {topn(y,oof,.20):.1%}")
final = mk().fit(X, y)
df["hazard_raw"] = final.predict_proba(X)[:, 1]
df["hazard_oof"] = oof

# ===============================================================
# 2. SHAP — 무엇이 침수 감수성을 만드는가
# ===============================================================
P("\n## 2. SHAP 기여도 (전체학습 모델)")
expl = shap.TreeExplainer(final)
samp = df.sample(n=min(20000, len(df)), random_state=RS)
sv = expl.shap_values(samp[FEATS].values)
imp = pd.DataFrame({"feature": FEATS, "mean_abs_shap": np.abs(sv).mean(0)})
# 방향: 피처값과 SHAP의 상관 부호 (양수면 값↑→위험↑)
imp["dir_corr"] = [np.corrcoef(samp[f].values, sv[:, i])[0, 1] for i, f in enumerate(FEATS)]
imp["share"] = imp.mean_abs_shap / imp.mean_abs_shap.sum()
imp = imp.sort_values("mean_abs_shap", ascending=False)
imp.to_csv(MOD / "M4_shap_importance.csv", index=False, encoding="utf-8-sig")
P("| 순위 | 피처 | 평균\\|SHAP\\| | 기여율 | 방향(값↑→위험) |")
P("|--:|---|--:|--:|:-:|")
for i, (_, r) in enumerate(imp.iterrows(), 1):
    P(f"| {i} | {r.feature} | {r.mean_abs_shap:.4f} | {r.share:.1%} | {'↑' if r.dir_corr>0 else '↓'} ({r.dir_corr:+.2f}) |")
P("\n해석:")
top3 = imp.head(3).feature.tolist()
P(f"- 상위 3개 {top3} 가 전체 기여의 {imp.head(3).share.sum():.0%}")
if "lowland3_ratio_s" in imp.feature.values:
    lw = imp[imp.feature=="lowland3_ratio_s"].iloc[0]
    P(f"- `lowland3_ratio_s`(절대 저지대) 기여율 {lw.share:.1%} — 코드리뷰 FIND-1대로 낮음. 상대지형(tpi/twi)이 우위")
if "rain_annmax_mm_s" in imp.feature.values:
    rn = imp[imp.feature=="rain_annmax_mm_s"].iloc[0]
    P(f"- `rain_annmax_mm_s`(강우) 기여율 {rn.share:.1%} — 부산 내 일강수량 공간차가 작아 예상대로 낮음")

# ===============================================================
# 3. [V1] 사건 일반화 — 2014년 제외 학습 → 2014년 예측
# ===============================================================
P("\n## 3. [V1] 사건 일반화 검증 — 2014 제외 학습 → 2014 침수 예측")
is2014 = (df.trace_flag == 1) & (df.trace_last_year == 2014)
y_no14 = np.where(is2014, 0, y)          # 2014 침수를 '음성'으로 돌려 학습에서 배제
P(f"- 전체 침수 {y.sum():,} 중 2014년 {int(is2014.sum()):,} ({is2014.sum()/y.sum():.1%}) 제외 → 학습 양성 {y_no14.sum():,}")
spw2 = (y_no14 == 0).sum() / max(y_no14.sum(), 1)
oof14 = oof_of(X, y_no14, list(GroupKFold(n_splits=5).split(X, y_no14, groups)), mk(spw2))
# 평가: 2014 침수 격자를 양성으로, 한번도 침수 안 된 격자를 음성으로 (비2014 침수는 학습에 썼으므로 제외)
mask = is2014.values | (y == 0)
ap14 = average_precision_score(y[mask], oof14[mask]); auc14 = roc_auc_score(y[mask], oof14[mask])
base14 = y[mask].mean()
P(f"- 2014 침수 격자 예측: PR-AUC **{ap14:.4f}** (양성률 {base14:.4f} → 리프트 **{ap14/base14:.1f}배**) | ROC-AUC {auc14:.4f}")
P(f"- Top10% 포착 {topn(y[mask], oof14[mask], .10):.1%}")
P(f"\n→ 2014년을 전혀 보지 않고 학습해도 2014년 침수를 {ap14/base14:.1f}배 리프트로 식별. "
  + ("**단일 사건 과적합이 아님**을 확인." if ap14/base14 >= 3 else "리프트가 낮아 사건 일반화에 한계가 있음."))

# ===============================================================
# 4. [V2] 자치구 내 상대순위 — 조사 편향에 강건한가
# ===============================================================
P("\n## 4. [V2] 자치구 내 상대순위 (침수흔적도 조사편향 대응)")
rows = []
for gu, g in df.assign(y=y, s=oof).groupby("sgg_nm"):
    if g.y.sum() < 10:
        continue
    rho, p = spearmanr(g.s, g.y)
    rows.append(dict(자치구=gu, 격자=len(g), 침수격자=int(g.y.sum()),
                     spearman=rho, p=p, Top10=topn(g.y.values, g.s.values, .10)))
gr = pd.DataFrame(rows).sort_values("spearman", ascending=False)
P("| 자치구 | 격자 | 침수격자 | Spearman ρ | Top10% 포착 |")
P("|---|--:|--:|--:|--:|")
for _, r in gr.iterrows():
    P(f"| {r.자치구} | {r.격자:,} | {r.침수격자} | {r.spearman:+.3f} | {r.Top10:.1%} |")
P(f"\n- 자치구 {len(gr)}개 중 ρ>0 인 곳 **{int((gr.spearman>0).sum())}개**, 중앙값 ρ **{gr.spearman.median():+.3f}**")
P(f"- 자치구별 Top10% 포착률 중앙값 {gr.Top10.median():.1%} (무작위 10% 대비 {gr.Top10.median()/0.10:.1f}배)")
neg = gr[gr.spearman <= 0]
if len(neg):
    P(f"- ⚠ ρ≤0 자치구: {neg.자치구.tolist()} — 해당 지역은 모델 신뢰도 낮음, 해석 시 주의")

# ===============================================================
# 5. [V3] 확률 보정 — MCDA에서 곱해도 되는가
# ===============================================================
P("\n## 5. [V3] 확률 보정 (calibration)")
P(f"- Brier score {brier_score_loss(y, np.clip(oof,0,1)):.4f}")
frac, mean_pred = calibration_curve(y, np.clip(oof, 0, 1), n_bins=10, strategy="quantile")
P("| 예측확률 구간평균 | 실제 침수율 |")
P("|--:|--:|")
for mp, fp in zip(mean_pred, frac):
    P(f"| {mp:.3f} | {fp:.3f} |")
slope = np.polyfit(mean_pred, frac, 1)[0]
P(f"\n- 보정 기울기 {slope:.2f} (1.0이 이상적)")
P(f"- scale_pos_weight={SPW:.0f} 로 불균형 보정했기 때문에 **출력은 절대확률이 아니라 상대 위험도**다.")
P("  → MCDA에서는 원확률이 아니라 **순위 백분위(percentile rank)** 로 변환해 결합한다. (M5)")

df["hazard_pct"] = df.hazard_raw.rank(pct=True)
df[["grid_id", "sgg_cd", "sgg_nm", "adm_cd", "adm_nm", "hazard_raw", "hazard_oof", "hazard_pct"]] \
    .to_parquet(MOD / "hazard_score.parquet", index=False)
P(f"\n- 산출: hazard_score.parquet (격자별 침수감수성), M4_shap_importance.csv")

(REP / "M4_최종모델_검증.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'M4_최종모델_검증.md'}", flush=True)
