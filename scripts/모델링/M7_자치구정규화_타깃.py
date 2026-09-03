# -*- coding: utf-8 -*-
"""
M7. 자치구 정규화 학습 — 조사편향 상쇄 시도

문제 (이슈 #13)
  침수흔적은 '실제 침수'가 아니라 '기록된 침수'라 지자체 조사강도 차이가 그대로 들어온다.
  기장군 886격자(전체의 31%) vs 부산진구 22격자. 토지피복을 추가해도 모델은
  `agri_ratio +0.69`(농업=위험)를 학습했는데, 이는 기장 농업지역의 과다 기록에서 온 것이다.

방법
  샘플 가중치로 **각 자치구가 양성/음성 모두에서 동등하게 기여**하도록 만든다.
    w_pos(i) = 1 / (그 자치구의 양성 수),  w_neg(i) = 1 / (그 자치구의 음성 수)
  → "기장에 침수가 많다"는 신호가 사라지고 각 자치구 **내부의 공간 패턴**만 학습된다.
  비교군으로 자치구 균등 언더샘플링도 함께 본다.

사전 등록한 성공 기준
  ① 자치구 내 Spearman ρ 중앙값 +0.265 → **+0.30 이상**
  ② 강서구 ρ +0.110 → **+0.20 이상**
  ③ SHAP 역전(lowland3 -0.97) 완화
  전역 PR-AUC 하락은 허용 (편향을 못 쓰게 되므로 당연)
  → 셋 중 둘 이상 미달이면 실패로 판정하고 타깃 자체를 교체(부산 안전ON 침수위 실측)한다.

- 입력: 04_모델/features_v3.parquet, M3_최종설정.json
- 출력: _리포트/M7_자치구정규화.md, 04_모델/oof_m7_*.npy
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

P("# M7. 자치구 정규화 학습 (조사편향 상쇄)")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

df = pd.read_parquet(MOD / "features_v3.parquet")
cfg = json.load(open(MOD / "M3_최종설정.json", encoding="utf-8"))
NEW = ["imperv_ratio", "agri_ratio", "paddy_ratio", "forest_ratio", "water_ratio", "road_ratio"]
FEATS = cfg["features"] + [c + "_s" for c in NEW]
X = df[FEATS].values
y = df.trace_flag.values
groups = df.sgg_cd.values
gkf = list(GroupKFold(n_splits=5).split(X, y, groups))
P(f"격자 {len(df):,} · 피처 {len(FEATS)} · 양성 {y.sum():,} ({y.mean():.2%})")

P("\n## 자치구별 침수흔적 편중 (문제의 크기)")
t = df.assign(y=y).groupby("sgg_nm").agg(격자=("grid_id", "size"), 양성=("y", "sum"))
t["양성률"] = t.양성 / t.격자
t["전체대비"] = t.양성 / t.양성.sum()
for nm, r in t.sort_values("양성", ascending=False).iterrows():
    P(f"  {nm:<7} 격자 {r.격자:>6,} | 양성 {int(r.양성):>4} ({r.양성률:.2%}) | 전체의 {r.전체대비:.1%}")
P(f"- 양성률 최대/최소 비율: {t.양성률.max()/t.양성률.min():.1f}배 (편중이 클수록 조사편향 의심)")

# ---- 샘플 가중치: 자치구 × 클래스 동등 기여 ----
w = np.ones(len(y), dtype=float)
for g in np.unique(groups):
    m = groups == g
    for cls in (0, 1):
        sel = m & (y == cls)
        n = sel.sum()
        if n:
            w[sel] = 1.0 / n
# 클래스 총합 균형 (기존 scale_pos_weight 역할을 가중치가 대신)
w[y == 1] *= (w[y == 0].sum() / w[y == 1].sum())
# ⚠ 스케일 정규화 필수: 1/n 그대로 두면 w~5e-5 라서 min_child_weight(=20) 조건을
#   어떤 분할도 만족하지 못해 트리가 단일 잎으로 붕괴한다(ROC 0.49, 예측 상수).
#   상대구조는 유지한 채 평균 가중치를 1로 맞춘다.
w *= len(y) / w.sum()
P(f"\n샘플 가중치: 자치구×클래스 동등 후 평균 1로 정규화")
P(f"  평균 {w.mean():.3f} | 양성 총가중 {w[y==1].sum():,.0f} / 음성 {w[y==0].sum():,.0f}")
P(f"  양성 가중 범위 {w[y==1].min():.3f}~{w[y==1].max():.3f} (기장 최소, 중구 최대여야 정상)")

def mk(spw=None):
    p = dict(cfg["params"]);
    return XGBClassifier(**p, scale_pos_weight=(spw if spw else 1.0), eval_metric="aucpr",
                         tree_method="hist", reg_lambda=1.0, n_jobs=-1,
                         random_state=RS, verbosity=0)

def oof_of(model, sw=None):
    o = np.zeros(len(y))
    for tr, te in gkf:
        m = clone(model)
        m.fit(X[tr], y[tr], sample_weight=(sw[tr] if sw is not None else None))
        o[te] = m.predict_proba(X[te])[:, 1]
    return o

def topn(yy, s, pct):
    k = max(1, int(len(s) * pct))
    o = s + np.random.RandomState(RS).rand(len(s)) * max(np.ptp(s), 1e-9) * 1e-9
    return yy[np.argpartition(-o, k - 1)[:k]].sum() / max(yy.sum(), 1)

def within_rho(score):
    rows = []
    for gu in df.sgg_nm.unique():
        m = (df.sgg_nm == gu).values
        if y[m].sum() < 10:
            continue
        rows.append((gu, spearmanr(score[m], y[m]).statistic, topn(y[m], score[m], .10)))
    return pd.DataFrame(rows, columns=["자치구", "rho", "top10"])

P("\n## 학습 비교")
runs = {}
runs["기준(가중 없음)"] = oof_of(mk(cfg["scale_pos_weight"]))
runs["자치구 정규화 가중"] = oof_of(mk(), w)
for nm, o in runs.items():
    np.save(MOD / f"oof_m7_{'base' if '기준' in nm else 'norm'}.npy", o)
    gr = within_rho(o)
    runs[nm] = dict(oof=o, ap=average_precision_score(y, o), auc=roc_auc_score(y, o),
                    t10=topn(y, o, .10), rho=gr.rho.median(), gr=gr)
    r = runs[nm]
    P(f"  {nm:<16} PR-AUC {r['ap']:.4f} | ROC {r['auc']:.4f} | 전역Top10% {r['t10']:.1%} | 자치구내 ρ중앙값 {r['rho']:+.3f}")

a, b = runs["기준(가중 없음)"], runs["자치구 정규화 가중"]
P("\n## 자치구별 ρ 변화")
mg = a["gr"].merge(b["gr"], on="자치구", suffixes=("_기준", "_정규화"))
mg["delta"] = mg.rho_정규화 - mg.rho_기준
P("| 자치구 | 기준 ρ | 정규화 ρ | 변화 | 정규화 Top10% |")
P("|---|--:|--:|--:|--:|")
for _, r in mg.sort_values("delta", ascending=False).iterrows():
    P(f"| {r.자치구} | {r.rho_기준:+.3f} | {r.rho_정규화:+.3f} | **{r.delta:+.3f}** | {r.top10_정규화:.1%} |")

P("\n## SHAP 방향 (역전 해소 여부)")
mdl = mk().fit(X, y, sample_weight=w)
samp = df.sample(n=min(20000, len(df)), random_state=RS)
sv = shap.TreeExplainer(mdl).shap_values(samp[FEATS].values)
imp = pd.DataFrame({"feature": FEATS, "shap": np.abs(sv).mean(0)})
imp["dir"] = [np.corrcoef(samp[f].values, sv[:, i])[0, 1] for i, f in enumerate(FEATS)]
imp["share"] = imp.shap / imp.shap.sum()
imp = imp.sort_values("shap", ascending=False)
imp.to_csv(MOD / "M7_shap_importance.csv", index=False, encoding="utf-8-sig")
for _, r in imp.head(10).iterrows():
    P(f"  {r.feature:<20} {r.share:5.1%} {'↑' if r['dir']>0 else '**↓**'} ({r['dir']:+.2f})")
PREV = {"lowland3_ratio_s": -0.97, "twi_s": -0.76, "rain_annmax_mm_s": -0.54, "agri_ratio_s": +0.69}
P("\n주요 피처 방향 변화 (M6 → M7):")
for f, before in PREV.items():
    now = float(imp[imp.feature == f]["dir"].iloc[0]) if f in imp.feature.values else np.nan
    P(f"  {f:<20} {before:+.2f} → **{now:+.2f}**")

# ---- 사전 등록 기준 판정 ----
P("\n## 사전 등록 기준 판정")
rho_med = b["rho"]
gang = float(mg[mg.자치구 == "강서구"].rho_정규화.iloc[0]) if (mg.자치구 == "강서구").any() else np.nan
low_now = float(imp[imp.feature == "lowland3_ratio_s"]["dir"].iloc[0])
c1 = rho_med >= 0.30
c2 = gang >= 0.20
c3 = low_now > -0.97 + 0.15
P(f"  ① 자치구내 ρ 중앙값 ≥ +0.300 : {rho_med:+.3f} → {'통과' if c1 else '미달'}")
P(f"  ② 강서구 ρ ≥ +0.200        : {gang:+.3f} → {'통과' if c2 else '미달'}")
P(f"  ③ lowland3 역전 완화(>-0.82): {low_now:+.3f} → {'통과' if c3 else '미달'}")
n_ok = sum([c1, c2, c3])
P(f"\n### 판정: {n_ok}/3 통과 → **{'성공 — 자치구 정규화 채택' if n_ok >= 2 else '실패 — 타깃 교체(부산 안전ON 침수위 실측) 필요'}**")
P(f"- 참고: 전역 PR-AUC {a['ap']:.4f} → {b['ap']:.4f} ({b['ap']/a['ap']-1:+.1%}) "
  "(하락은 예상된 결과 — 자치구별 기록량 차이를 더는 이용하지 않기 때문)")

(REP / "M7_자치구정규화.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'M7_자치구정규화.md'}", flush=True)
