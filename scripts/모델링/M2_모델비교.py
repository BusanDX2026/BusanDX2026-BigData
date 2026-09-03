# -*- coding: utf-8 -*-
"""
M2. 모델 비교 · 성능검사 · 검증설계 타당성 실험

무엇을 왜 비교하는가
  [실험1] 검증설계: 랜덤 K-fold vs 자치구 GroupKFold
      └ 공간 자기상관 때문에 랜덤 분할은 성능을 과대평가한다. 그 크기를 수치로 보인다.
  [실험2] 피처군: HAZARD(물리) vs HAZARD+EXPOSURE+CAPACITY(전체)
      └ 타깃이 '기록된 침수'라 인구를 넣으면 보고편향을 학습한다(코드리뷰 FIND-3).
        또 펌프장은 침수위험지에 설치되는 역인과라 순환논리를 만든다. 그 크기를 수치로 보인다.
  [실험3] 알고리즘: 상수/단일지표/MCDA동일가중/로지스틱/RF/XGBoost/LightGBM
      └ 왜 굳이 이 모델인가를 '더 단순한 대안 대비 얼마나 나은가'로 답한다.

평가지표 (양성 3.55% 불균형)
  - PR-AUC(Average Precision)  ← **주 지표**. 불균형에서 ROC-AUC는 낙관적으로 부풀려짐
  - ROC-AUC                    ← 관행상 병기
  - Top5%/Top10% 포착률         ← 정책 지표: 우선순위 상위 N%에 실제 침수의 몇 %가 잡히나
  - Brier score                ← 확률 보정 품질
  - fit/predict 시간           ← 코드 최적화 판단 근거

- 입력: 04_모델/features.parquet, feature_groups.json
- 출력: 04_모델/M2_모델비교.csv, _리포트/M2_모델비교.md
- 규약: random_state=42 고정
"""
import sys, io, json, time, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"
MOD = GG / "04_모델"
REP = GG / "_리포트"
RS = 42
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# M2. 모델 비교 · 성능검사")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

df = pd.read_parquet(MOD / "features.parquet")
GROUPS = json.load(open(MOD / "feature_groups.json", encoding="utf-8"))
HAZ = [c + "_s" for c in GROUPS["HAZARD"]]
ALL = [c + "_s" for c in GROUPS["HAZARD"] + GROUPS["EXPOSURE"] + GROUPS["CAPACITY"]]
y = df["trace_flag"].values
groups = df["sgg_cd"].values
P(f"격자 {len(df):,} · 양성 {y.sum():,} ({y.mean():.2%}) · 자치구 {len(np.unique(groups))}개")
P(f"HAZARD 피처 {len(HAZ)}: {HAZ}")
P(f"ALL 피처 {len(ALL)}\n")

# ---------------------------------------------------------------
# 평가 함수
# ---------------------------------------------------------------
_TIE = np.random.RandomState(RS).rand(len(pd.read_parquet(MOD / "features.parquet")))

def capture_rate(y_true, score, pct):
    """상위 pct% 격자에 실제 양성이 몇 % 포함되나 (정책 지표)

    ⚠ 동점(tie) 처리: 이진 지표(재해위험지구)나 5~95% 클리핑으로 상한에 몰린 지표(elev_min_s)는
      동점이 대량 발생한다. np.argsort는 동점을 '인덱스 순서'로 깨는데 인덱스는 격자 좌표순이라
      공간적으로 편향된 선택이 되어 포착률이 무작위보다도 낮게 나온다(관측된 버그).
      → 고정 시드 난수를 미세 가산해 동점을 무작위로 깬다(기대값 = 무작위 타이브레이크).
    """
    s = np.asarray(score, dtype=float)
    rng = _TIE[:len(s)]
    span = np.ptp(s)
    s = s + rng * (span if span > 0 else 1.0) * 1e-9
    k = max(1, int(len(s) * pct))
    idx = np.argpartition(-s, k - 1)[:k]
    return y_true[idx].sum() / max(y_true.sum(), 1)

def evaluate(y_true, score):
    return dict(
        PR_AUC=average_precision_score(y_true, score),
        ROC_AUC=roc_auc_score(y_true, score),
        Top5=capture_rate(y_true, score, 0.05),
        Top10=capture_rate(y_true, score, 0.10),
        Brier=brier_score_loss(y_true, np.clip(score, 0, 1)),
    )

def make_models(n_pos, n_neg):
    spw = n_neg / max(n_pos, 1)          # 불균형 보정
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RS),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=20,
            class_weight="balanced_subsample", n_jobs=-1, random_state=RS),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="aucpr",
            tree_method="hist", n_jobs=-1, random_state=RS, verbosity=0),
        "LightGBM": LGBMClassifier(
            n_estimators=400, num_leaves=31, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=spw, min_child_samples=30,
            n_jobs=-1, random_state=RS, verbose=-1),
    }

def cv_eval(X, y, groups, splitter, models, tag):
    """교차검증: 폴드별 예측을 모아 전체 지표 산출(OOF)"""
    rows = []
    for name, proto in models.items():
        oof = np.zeros(len(y)); t_fit = t_pred = 0.0
        for tr, te in splitter:
            from sklearn.base import clone
            mdl = clone(proto)
            t0 = time.perf_counter(); mdl.fit(X[tr], y[tr]); t_fit += time.perf_counter() - t0
            t0 = time.perf_counter(); oof[te] = mdl.predict_proba(X[te])[:, 1]; t_pred += time.perf_counter() - t0
        r = evaluate(y, oof); r.update(model=name, setting=tag, fit_s=round(t_fit, 2), pred_s=round(t_pred, 3))
        rows.append(r)
        P(f"  {name:<20} PR-AUC {r['PR_AUC']:.4f} | ROC {r['ROC_AUC']:.4f} | "
          f"Top5% {r['Top5']:.1%} | Top10% {r['Top10']:.1%} | fit {r['fit_s']}s")
    return rows

results = []

# ---------------------------------------------------------------
# 실험0. 단순 기준선 (모델이 이걸 못 이기면 쓸 이유가 없다)
# ---------------------------------------------------------------
P("## 실험0. 단순 기준선 (비학습)")
base = {
    "무작위(상수)": np.random.RandomState(RS).rand(len(df)),
    "단일지표: 표고만(elev_min_s)": df["elev_min_s"].values,
    "단일지표: TPI만(tpi_s)": df["tpi_s"].values,
    "MCDA 동일가중(HAZARD 6개 평균)": df[HAZ].mean(axis=1).values,
    "행정 baseline: 재해위험지구 지정여부": df["hazdist_flood_active"].values.astype(float),
}
for nm, sc in base.items():
    r = evaluate(y, sc); r.update(model=nm, setting="baseline", fit_s=0, pred_s=0)
    results.append(r)
    P(f"  {nm:<34} PR-AUC {r['PR_AUC']:.4f} | ROC {r['ROC_AUC']:.4f} | Top5% {r['Top5']:.1%} | Top10% {r['Top10']:.1%}")

# ---------------------------------------------------------------
# 실험1+2. 검증설계 × 피처군 × 알고리즘
# ---------------------------------------------------------------
n_pos, n_neg = int(y.sum()), int((y == 0).sum())
models = make_models(n_pos, n_neg)
skf = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=RS).split(df[HAZ].values, y))
gkf = list(GroupKFold(n_splits=5).split(df[HAZ].values, y, groups))

for feats, fname in [(HAZ, "HAZARD"), (ALL, "ALL")]:
    X = df[feats].values
    for splitter, sname in [(skf, "랜덤5-fold"), (gkf, "자치구GroupKFold5")]:
        P(f"\n## {fname} 피처 × {sname}")
        results += cv_eval(X, y, groups, splitter, models, f"{fname}/{sname}")

res = pd.DataFrame(results)[["setting", "model", "PR_AUC", "ROC_AUC", "Top5", "Top10", "Brier", "fit_s", "pred_s"]]
res.to_csv(MOD / "M2_모델비교.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------
# 해석
# ---------------------------------------------------------------
P("\n---\n## 결과 해석")

def get(setting, model, col="PR_AUC"):
    s = res[(res.setting == setting) & (res.model == model)][col]
    return float(s.iloc[0]) if len(s) else np.nan

P("\n### 실험1. 공간 자기상관 — 랜덤 분할은 성능을 얼마나 부풀리나")
P("| 피처군 | 모델 | 랜덤5-fold PR-AUC | 자치구 GroupKFold PR-AUC | 과대평가 |")
P("|---|---|--:|--:|--:|")
for fname in ["HAZARD", "ALL"]:
    for mn in models:
        a = get(f"{fname}/랜덤5-fold", mn); b = get(f"{fname}/자치구GroupKFold5", mn)
        P(f"| {fname} | {mn} | {a:.4f} | {b:.4f} | **{(a/b-1)*100:+.1f}%** |")
P("\n→ 랜덤 분할은 같은 침수 폴리곤이 학습·검증에 쪼개져 들어가 성능이 부풀려진다. "
  "**자치구 GroupKFold 수치만 신뢰한다.**")

P("\n### 실험2. 보고편향·역인과 — 인구/펌프를 넣으면 성능이 오르지만 그건 물리가 아니다")
P("| 모델 | HAZARD PR-AUC | ALL PR-AUC | 증가분 |")
P("|---|--:|--:|--:|")
for mn in models:
    a = get("HAZARD/자치구GroupKFold5", mn); b = get("ALL/자치구GroupKFold5", mn)
    P(f"| {mn} | {a:.4f} | {b:.4f} | **{(b/a-1)*100:+.1f}%** |")
P("\n→ ALL이 더 높아도 채택하지 않는다. 인구는 '침수가 기록될 확률'을 올릴 뿐 물이 모이는 원인이 아니고, "
  "펌프장은 침수위험지에 설치되므로 역인과다. **선제대응구역은 노출을 MCDA에서 명시적으로 곱해야 정책 해석이 가능하다.**")

best = res[res.setting == "HAZARD/자치구GroupKFold5"].sort_values("PR_AUC", ascending=False)
P("\n### 실험3. 알고리즘 선택 (HAZARD × 자치구 GroupKFold 기준)")
P("| 순위 | 모델 | PR-AUC | ROC-AUC | Top5% 포착 | Top10% 포착 | fit(s) |")
P("|--:|---|--:|--:|--:|--:|--:|")
for i, (_, r) in enumerate(best.iterrows(), 1):
    P(f"| {i} | {r.model} | **{r.PR_AUC:.4f}** | {r.ROC_AUC:.4f} | {r.Top5:.1%} | {r.Top10:.1%} | {r.fit_s} |")
bl = res[res.setting == "baseline"].sort_values("PR_AUC", ascending=False)
P("\n기준선 대비:")
for _, r in bl.iterrows():
    P(f"- {r.model}: PR-AUC {r.PR_AUC:.4f} | Top10% {r.Top10:.1%}")
top = best.iloc[0]
mcda = bl[bl.model.str.contains("MCDA")].iloc[0]
P(f"\n→ 최고 모델 **{top.model}** PR-AUC {top.PR_AUC:.4f} vs MCDA 동일가중 {mcda.PR_AUC:.4f} "
  f"(**{(top.PR_AUC/mcda.PR_AUC-1)*100:+.1f}%**) — 학습이 단순 가중합보다 나은 만큼만 정당화된다.")
P(f"- 양성률 {y.mean():.2%} 대비 PR-AUC 리프트 **{top.PR_AUC/y.mean():.1f}배**")

P("\n### 성능·비용")
P("| 모델 | fit(s, 5폴드 합) | predict(s) | 상대비용 |")
P("|---|--:|--:|--:|")
mn_fit = res[res.fit_s > 0].fit_s.min()
for _, r in best.iterrows():
    P(f"| {r.model} | {r.fit_s} | {r.pred_s} | {r.fit_s/mn_fit:.1f}배 |")
P(f"\n- 산출: {(MOD/'M2_모델비교.csv').relative_to(ROOT)}")

(REP / "M2_모델비교.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'M2_모델비교.md'}", flush=True)
