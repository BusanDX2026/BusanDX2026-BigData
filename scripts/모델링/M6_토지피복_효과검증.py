# -*- coding: utf-8 -*-
"""
M6. 토지피복(불투수면적) 추가 효과 검증

가설
  H1 성능: 불투수면적률은 내수침수 유출량의 직접 원인이므로 PR-AUC가 오른다.
  H2 편향: 모델이 강서 삼각주를 "저지대=안전"으로 역방향 학습한 원인(이슈 #5)은
           '저지대인데 농경지'를 구분 못해서였다. 농업지역 비율을 주면
           ① 강서 자치구 내 Spearman ρ 가 오르고
           ② lowland3/twi/rain 의 SHAP 방향 역전이 완화된다.

- 입력: 04_모델/features_v2.parquet, 02_레이어별/토지피복_grid.parquet, M3_최종설정.json
- 출력: 04_모델/features_v3.parquet, _리포트/M6_토지피복효과.md
"""
import sys, io, json, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
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

P("# M6. 토지피복 추가 효과 검증")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

df = pd.read_parquet(MOD / "features_v2.parquet")
lc = pd.read_parquet(GG / "02_레이어별" / "토지피복_grid.parquet")
df = df.merge(lc, on="grid_id", how="left")
NEW = ["imperv_ratio", "agri_ratio", "paddy_ratio", "forest_ratio", "water_ratio", "road_ratio"]
for c in NEW + ["grass_ratio", "bare_ratio"]:
    df[c] = df[c].fillna(0)
# 면적비는 이미 0~1 → 정규화 불필요, 방향만 확인
for c in NEW:
    df[c + "_s"] = df[c].clip(0, 1)
df.to_parquet(MOD / "features_v3.parquet", index=False)

cfg = json.load(open(MOD / "M3_최종설정.json", encoding="utf-8"))
BASE = cfg["features"]; PARAMS = cfg["params"]; SPW = cfg["scale_pos_weight"]
PLUS = BASE + [c + "_s" for c in NEW]
y = df.trace_flag.values; groups = df.sgg_cd.values
gkf = list(GroupKFold(n_splits=5).split(df, y, groups))
P(f"격자 {len(df):,} · 양성 {y.sum():,} ({y.mean():.2%})")
P(f"기존 피처 {len(BASE)} → 토지피복 6개 추가 → {len(PLUS)}\n")

P("## 0. 방향 검증 (침수흔적 격자에서 높아야 위험 방향)")
for c in NEW:
    a = df.loc[y == 1, c].mean(); z = df.loc[y == 0, c].mean()
    P(f"  {'OK ' if a > z else '역전'} {c:<14} 침수O {a:.3f} | 침수X {z:.3f} | {a-z:+.3f}")

def mk():
    return XGBClassifier(**PARAMS, scale_pos_weight=SPW, eval_metric="aucpr",
                         tree_method="hist", reg_lambda=1.0, n_jobs=-1,
                         random_state=RS, verbosity=0)

def oof_of(X):
    o = np.zeros(len(y))
    for tr, te in gkf:
        m = clone(mk()); m.fit(X[tr], y[tr]); o[te] = m.predict_proba(X[te])[:, 1]
    return o

def topn(yy, s, pct):
    k = max(1, int(len(s) * pct))
    o = s + np.random.RandomState(RS).rand(len(s)) * max(np.ptp(s), 1e-9) * 1e-9
    return yy[np.argpartition(-o, k - 1)[:k]].sum() / max(yy.sum(), 1)

P("\n## 1. [H1] 성능 효과 (자치구 GroupKFold5)")
res = {}
for feats, nm in [(BASE, "기존 9피처"), (PLUS, "＋토지피복 6 (15)")]:
    oof = oof_of(df[feats].values)
    res[nm] = dict(oof=oof, ap=average_precision_score(y, oof), auc=roc_auc_score(y, oof),
                   t10=topn(y, oof, .10), t5=topn(y, oof, .05))
    r = res[nm]
    P(f"  {nm:<20} PR-AUC {r['ap']:.4f} | ROC {r['auc']:.4f} | Top5% {r['t5']:.1%} | Top10% {r['t10']:.1%}")
a, b = res["기존 9피처"], res["＋토지피복 6 (15)"]
P(f"\n→ PR-AUC **{(b['ap']/a['ap']-1):+.1%}** · Top10% 포착 {a['t10']:.1%} → **{b['t10']:.1%}**")

P("\n## 2. [H2] 자치구 내 상대순위 변화 (이슈 #5 해소 여부)")
rows = []
for gu in df.sgg_nm.unique():
    m = df.sgg_nm == gu
    if y[m.values].sum() < 10:
        continue
    r0 = spearmanr(a["oof"][m.values], y[m.values]).statistic
    r1 = spearmanr(b["oof"][m.values], y[m.values]).statistic
    rows.append(dict(자치구=gu, before=r0, after=r1, delta=r1 - r0))
gr = pd.DataFrame(rows).sort_values("delta", ascending=False)
P("| 자치구 | 기존 ρ | 토지피복 후 ρ | 변화 |")
P("|---|--:|--:|--:|")
for _, r in gr.iterrows():
    P(f"| {r.자치구} | {r.before:+.3f} | {r.after:+.3f} | **{r.delta:+.3f}** |")
P(f"\n- 중앙값 ρ {gr.before.median():+.3f} → **{gr.after.median():+.3f}**")
gs = gr[gr.자치구 == "강서구"]
if len(gs):
    r = gs.iloc[0]
    P(f"- **강서구 ρ {r.before:+.3f} → {r.after:+.3f} ({r.delta:+.3f})** "
      + ("← 이슈 #5 완화 확인" if r.delta > 0.05 else "← 개선 미미, 이슈 #5 잔존"))

P("\n## 3. [H2] SHAP 방향 역전 해소 여부")
final = mk().fit(df[PLUS].values, y)
expl = shap.TreeExplainer(final)
samp = df.sample(n=min(20000, len(df)), random_state=RS)
sv = expl.shap_values(samp[PLUS].values)
imp = pd.DataFrame({"feature": PLUS, "mean_abs_shap": np.abs(sv).mean(0)})
imp["dir"] = [np.corrcoef(samp[f].values, sv[:, i])[0, 1] for i, f in enumerate(PLUS)]
imp["share"] = imp.mean_abs_shap / imp.mean_abs_shap.sum()
imp = imp.sort_values("mean_abs_shap", ascending=False)
imp.to_csv(MOD / "M6_shap_importance.csv", index=False, encoding="utf-8-sig")
P("| 피처 | 기여율 | 방향 |")
P("|---|--:|:-:|")
for _, r in imp.iterrows():
    P(f"| {r.feature} | {r.share:.1%} | {'↑' if r['dir']>0 else '**↓**'} ({r['dir']:+.2f}) |")
prev = {"lowland3_ratio_s": -0.96, "twi_s": -0.77, "rain_annmax_mm_s": -0.51}
P("\n역전 3인방 변화 (음수일수록 '저지대=안전' 오학습):")
for f, before in prev.items():
    now = float(imp[imp.feature == f]["dir"].iloc[0]) if f in imp.feature.values else np.nan
    P(f"  {f:<20} {before:+.2f} → **{now:+.2f}** ({'완화' if now > before else '악화/유지'})")

P(f"\n- 산출: features_v3.parquet, M6_shap_importance.csv")
(REP / "M6_토지피복효과.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'M6_토지피복효과.md'}", flush=True)
