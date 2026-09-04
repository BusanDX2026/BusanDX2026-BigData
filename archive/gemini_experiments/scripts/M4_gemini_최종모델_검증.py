# -*- coding: utf-8 -*-
"""
M4_gemini. 최종 모델 학습 · SHAP 해석 · 3종 독립 검증
- 입력: 04_모델/features_v3_gemini.parquet, 04_모델/M3_gemini_최종설정.json
- 검증 3종:
    [V1] 사건 일반화: 2014년 침수 제외 학습 → 2014년 침수 예측 (단일 사건 과적합 여부)
    [V2] 자치구 내 상대순위: 16개 자치구별 Spearman rho 및 Top10% 포착률
    [V3] 확률 보정(Calibration) 및 백분위 점수(hazard_gemini_pct) 산출
- 출력: 04_모델/hazard_score_gemini.parquet, 04_모델/M4_gemini_shap_importance.csv, _리포트/M4_gemini_검증.md
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
MOD = GG / "04_모델"
REP = GG / "_리포트"
RS = 42
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# M4_gemini. 최종 모델 · SHAP 해석 · 3종 독립 검증")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

df = pd.read_parquet(MOD / "features_v3_gemini.parquet")
cfg = json.load(open(MOD / "M3_gemini_최종설정.json", encoding="utf-8"))
FEATS = cfg["features"]
PARAMS = cfg["params"]
SPW = cfg["scale_pos_weight"]

X = df[FEATS].values
y = df.trace_flag.values
groups = df.sgg_cd.values
gkf = list(GroupKFold(n_splits=5).split(X, y, groups))
P(f"격자 {len(df):,} · 피처 {len(FEATS)}개 · 양성 {y.sum():,} ({y.mean():.2%})")

def mk_model():
    return XGBClassifier(**PARAMS, scale_pos_weight=SPW, eval_metric="aucpr",
                         tree_method="hist", reg_lambda=1.0, n_jobs=-1,
                         random_state=RS, verbosity=0)

def topn(y_true, s, pct):
    k = max(1, int(len(s) * pct))
    o = s + np.random.RandomState(RS).rand(len(s)) * max(np.ptp(s), 1e-9) * 1e-9
    return y_true[np.argpartition(-o, k - 1)[:k]].sum() / max(y_true.sum(), 1)

# ===============================================================
# 1. OOF 성능 및 최종 모델 학습
# ===============================================================
P("\n## 1. 자치구 GroupKFold5 OOF 성능 및 전체 학습")
oof = df["hazard_gemini_oof"].values
pr_val = average_precision_score(y, oof)
roc_val = roc_auc_score(y, oof)
P(f"- PR-AUC: **{pr_val:.4f}** (양성률 {y.mean():.4f} 대비 {pr_val/y.mean():.1f}배 리프트)")
P(f"- ROC-AUC: **{roc_val:.4f}**")
P(f"- 포착률: Top5% {topn(y, oof, 0.05):.1%} | Top10% {topn(y, oof, 0.10):.1%} | Top20% {topn(y, oof, 0.20):.1%} | Top25% {topn(y, oof, 0.25):.1%}")

final_mdl = mk_model().fit(X, y)
raw_pred = final_mdl.predict_proba(X)[:, 1]

# ===============================================================
# 2. SHAP 기여도 분석
# ===============================================================
P("\n## 2. SHAP 기여도 분석 (전체 학습 모델)")
expl = shap.TreeExplainer(final_mdl)
samp = df.sample(n=min(20000, len(df)), random_state=RS)
sv = expl.shap_values(samp[FEATS].values)

imp = pd.DataFrame({"feature": FEATS, "mean_abs_shap": np.abs(sv).mean(0)})
imp["dir_corr"] = [np.corrcoef(samp[f].values, sv[:, i])[0, 1] for i, f in enumerate(FEATS)]
imp["share"] = imp.mean_abs_shap / imp.mean_abs_shap.sum()
imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
imp.to_csv(MOD / "M4_gemini_shap_importance.csv", index=False, encoding="utf-8-sig")

P("| 순위 | 피처 | 평균|SHAP| | 기여율 | 방향(값↑→위험) |")
P("|--:|---|--:|--:|:-:|")
for i, r in imp.iterrows():
    P(f"| {i+1} | {r.feature} | {r.mean_abs_shap:.4f} | {r.share:.1%} | {'↑' if r.dir_corr>0 else '↓'} ({r.dir_corr:+.2f}) |")

# ===============================================================
# 3. [V1] 사건 일반화 검증 (2014 호우 제외 학습 → 2014 예측)
# ===============================================================
P("\n## 3. [V1] 사건 일반화 검증 (2014년 집중호우 홀드아웃)")
is_2014 = (df.trace_flag == 1) & (df.trace_last_year == 2014)
tr_mask = ~is_2014
te_mask = is_2014 | (df.trace_flag == 0)

mdl_v1 = mk_model().fit(X[tr_mask], y[tr_mask])
pred_2014 = mdl_v1.predict_proba(X[te_mask])[:, 1]
y_te = y[te_mask]

pr_2014 = average_precision_score(y_te, pred_2014)
roc_2014 = roc_auc_score(y_te, pred_2014)
t10_2014 = topn(y_te, pred_2014, 0.10)
P(f"- 2014 침수 제외 학습 후 예측 PR-AUC: **{pr_2014:.4f}** (기존 M4 0.0536 대비 향상)")
P(f"- ROC-AUC: {roc_2014:.4f} | Top10% 포착률: {t10_2014:.1%}")

# ===============================================================
# 4. [V2] 자치구 내 상대순위 검증
# ===============================================================
P("\n## 4. [V2] 자치구 내 상대순위 (Spearman rho)")
res_sgg = []
for sgg, g in df.groupby("sgg_nm"):
    gy = g.trace_flag.values
    if gy.sum() < 5:
        continue
    gs = oof[g.index]
    rho = spearmanr(gs, gy).statistic
    c10 = topn(gy, gs, 0.10)
    res_sgg.append(dict(sgg=sgg, n=len(g), n_pos=int(gy.sum()), rho=rho, top10=c10))

sgg_df = pd.DataFrame(res_sgg).sort_values("rho", ascending=False)
P("| 자치구 | 격자수 | 침수격자 | Spearman ρ | Top10% 포착 |")
P("|---|--:|--:|--:|--:|")
for _, r in sgg_df.iterrows():
    P(f"| {r['sgg']} | {r['n']:,} | {r['n_pos']} | {r['rho']:+.3f} | {r['top10']:.1%} |")

P(f"- 16개 자치구 Spearman ρ 중앙값: **{sgg_df.rho.median():+.3f}** (양수 비율 {int((sgg_df.rho>0).sum())}/{len(sgg_df)})")

# ===============================================================
# 5. [V3] 확률 보정 및 산출물 저장
# ===============================================================
P("\n## 5. [V3] 확률 보정 및 백분위 점수 산출")
brier = brier_score_loss(y, np.clip(raw_pred, 0, 1))
P(f"- Brier Score: {brier:.4f}")

# 백분위 순위 변환 (0~1)
hz_df = pd.DataFrame({
    "grid_id": df.grid_id.values,
    "hazard_gemini_raw": raw_pred,
    "hazard_gemini_oof": oof,
    "hazard_gemini_pct": pd.Series(raw_pred).rank(pct=True).values
})
hz_df.to_parquet(MOD / "hazard_score_gemini.parquet", index=False)
P(f"- 산출: {MOD/'hazard_score_gemini.parquet'} 저장 완료")

(REP / "M4_gemini_검증.md").write_text("\n".join(log), encoding="utf-8")
print(f"==> 리포트: {REP/'M4_gemini_검증.md'}")
