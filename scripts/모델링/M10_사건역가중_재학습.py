# -*- coding: utf-8 -*-
"""
M10. 침수흔적 타깃 '사건 크기 역가중' 재학습

문제
  침수흔적 2,893격자 중 57%가 2014-08-25 단일사건. XGBoost가 사실상 2014 재현기로 학습됨.
  → 비2014·재발형 침수의 재현율이 눌리고, 2014 홀드아웃 리프트가 2.5배로 낮게 나옴.

처리
  양성 격자 가중치 = f(사건 크기).  사건 = trace_last_year (격자 단위로 가진 유일 키).
    - W0 uniform          : 현행 M9
    - W1 1/n_event        : 사건별 총 영향력 균등 (2014 대폭 축소)
    - W2 1/sqrt(n_event)  : 완충형
    - W3 recurrence       : n_events(재발횟수) 비례 — 상습성 강조
  각 양성가중은 '양성 평균=1'로 정규화 후, 클래스 균형(neg/pos≈27)을 곱해 반영.
  음성 가중 = 1.  (scale_pos_weight 대신 sample_weight로 균형 → M7 교훈: 미소가중 붕괴 방지)

평가 (자치구 GroupKFold5 OOF)
  - PR-AUC(전체) / 재현율@상위10·15·25% : 타깃 {전체·비2014·재발형}
  - 홀드아웃 리프트 : 2014(극한) vs 2020(대형)
출력: _리포트/M10_사건역가중.md, 04_모델/M10_비교.csv
"""
import sys, io, json, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"; MOD = GG / "04_모델"; REP = GG / "_리포트"; LAY = GG / "02_레이어별"
RS = 42
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# M10. 침수흔적 사건 크기 역가중 재학습")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

df = pd.read_parquet(MOD / "features_v4.parquet")
act = pd.read_parquet(LAY / "침수흔적_활성화강우_격자.parquet")[["grid_id", "n_events"]]
df = df.merge(act, on="grid_id", how="left")
df["n_events"] = df["n_events"].fillna(0)
cfg = json.load(open(MOD / "M9_최종설정.json", encoding="utf-8"))
FEATS, PARAMS = cfg["features"], cfg["params"]
y = df.trace_flag.values.astype(int)
groups = df.sgg_cd.values
pos = y == 1
neg_pos_ratio = (~pos).sum() / pos.sum()
P(f"양성 {pos.sum():,} / 음성 {(~pos).sum():,}  (neg/pos = {neg_pos_ratio:.1f})")

# 사건 = trace_last_year
ev = df.loc[pos, "trace_last_year"].fillna(-1)
ev_size = ev.map(ev.value_counts())
P("\n사건(trace_last_year)별 양성 격자수:")
for yr, c in ev.value_counts().sort_index().items():
    P(f"  {int(yr)}: {c:,} ({c/pos.sum():.0%})")

def norm_pos_w(w):
    w = np.asarray(w, float)
    return w / w.mean()          # 양성 평균 = 1

WSETS = {
    "W0 uniform(현행M9)": np.ones(pos.sum()),
    "W1 1/n_event":       norm_pos_w(1.0 / ev_size.values),
    "W2 1/sqrt(n_event)": norm_pos_w(1.0 / np.sqrt(ev_size.values)),
    "W3 recurrence":      norm_pos_w(np.clip(df.loc[pos, "n_events"].values, 1, None)),
}

gkf = list(GroupKFold(5).split(df, y, groups))
X = df[FEATS].values
non2014 = (df.trace_last_year != 2014).values & pos
recur = (df.n_events >= 2).values & pos
is2020 = (df.trace_last_year == 2020).values & pos
is2014 = (df.trace_last_year == 2014).values & pos

def make():
    return XGBClassifier(**PARAMS, eval_metric="aucpr", tree_method="hist", reg_lambda=1.0,
                         n_jobs=-1, random_state=RS, verbosity=0)

def oof_weighted(pos_w):
    w = np.ones(len(y))
    w[pos] = pos_w * neg_pos_ratio        # 클래스 균형을 가중에 반영
    o = np.zeros(len(y))
    for tr, te in gkf:
        m = clone(make()); m.fit(X[tr], y[tr], sample_weight=w[tr])
        o[te] = m.predict_proba(X[te])[:, 1]
    return o

def recall_at(o, target_mask, pct):
    k = int(len(o) * pct)
    j = o + np.random.RandomState(RS).rand(len(o)) * 1e-9
    top = np.zeros(len(o), bool); top[np.argpartition(-j, k - 1)[:k]] = True
    return (top & target_mask).sum() / target_mask.sum()

def holdout_lift(pos_w, held_mask):
    """해당 가중을 실제로 적용해 홀드아웃 사건을 예측. (구버전은 pos_w 를 버려
    4개 가중이 동일 결과를 내던 버그 — 코드리뷰 #3)"""
    y0 = y.copy(); y0[held_mask] = 0
    p0 = y0 == 1
    # 남은 양성에 해당하는 가중만 취해 평균 1로 재정규화
    keep = (~held_mask)[pos]
    pw = np.asarray(pos_w, float)[keep]
    pw = pw / pw.mean()
    w = np.ones(len(y0))
    w[p0] = pw * ((y0 == 0).sum() / max(p0.sum(), 1))
    o = np.zeros(len(y0))
    for tr, te in GroupKFold(5).split(df, y0, groups):
        m = clone(make()); m.fit(X[tr], y0[tr], sample_weight=w[tr]); o[te] = m.predict_proba(X[te])[:, 1]
    m = held_mask | (y == 0)
    ap = average_precision_score(y[m], o[m]); b = y[m].mean()
    return ap / b, recall_at(o, held_mask, 0.10)

P("\n" + "=" * 78)
P("  가중별 성능 (자치구 GroupKFold5 OOF)")
P("=" * 78)
rows = []
oofs = {}
for nm, pw in WSETS.items():
    o = oof_weighted(pw); oofs[nm] = o
    ap = average_precision_score(y, o)
    r_all10, r_all25 = recall_at(o, pos, .10), recall_at(o, pos, .25)
    r_n14_10, r_n14_25 = recall_at(o, non2014, .10), recall_at(o, non2014, .25)
    r_rec10, r_rec15, r_rec25 = recall_at(o, recur, .10), recall_at(o, recur, .15), recall_at(o, recur, .25)
    rows.append(dict(가중=nm, PR_AUC=ap,
                     전체_top10=r_all10, 전체_top25=r_all25,
                     비2014_top10=r_n14_10, 비2014_top25=r_n14_25,
                     재발_top10=r_rec10, 재발_top15=r_rec15, 재발_top25=r_rec25))
    P(f"\n[{nm}]  PR-AUC {ap:.4f}")
    P(f"   재현율@상위10% : 전체 {r_all10:.1%} | 비2014 {r_n14_10:.1%} | 재발형 {r_rec10:.1%}")
    P(f"   재현율@상위25% : 전체 {r_all25:.1%} | 비2014 {r_n14_25:.1%} | 재발형 {r_rec25:.1%}")
    P(f"   재발형 @상위15% : {r_rec15:.1%}")

P("\n" + "=" * 78)
P("  홀드아웃 리프트 (해당 가중으로 학습, 사건 제외 후 예측)")
P("=" * 78)
for nm, pw in WSETS.items():
    l14, c14 = holdout_lift(pw, is2014)
    l20, c20 = holdout_lift(pw, is2020)
    P(f"  [{nm}]  2014: 리프트 {l14:.1f}배·포착 {c14:.1%}   |   2020: 리프트 {l20:.1f}배·포착 {c20:.1%}")
    for r in rows:
        if r["가중"] == nm:
            r["홀드2014_리프트"], r["홀드2020_리프트"] = round(l14, 2), round(l20, 2)

comp = pd.DataFrame(rows)
comp.to_csv(MOD / "M10_비교.csv", index=False, encoding="utf-8-sig")
P(f"\n저장: {MOD/'M10_비교.csv'}")

# 최선 가중 판정: 재발형@상위15% + 비2014@상위25% 합이 최대 (전체 PR-AUC 5% 이내 유지 조건)
comp["_score"] = comp["재발_top15"] + comp["비2014_top25"]
ok = comp[comp.PR_AUC >= comp.PR_AUC.max() * 0.95]
best = ok.sort_values("_score", ascending=False).iloc[0]
P(f"\n→ 권장 가중: **{best['가중']}**  "
  f"(재발형@15% {best['재발_top15']:.1%}, 비2014@25% {best['비2014_top25']:.1%}, "
  f"PR-AUC {best['PR_AUC']:.4f}, 2020 홀드아웃 {best.get('홀드2020_리프트','-')}배)")
P("  ※ 현행 W0 대비 개선폭이 <2%p면 재학습 실익 없음 → M9 유지 판단")

(REP / "M10_사건역가중.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'M10_사건역가중.md'}", flush=True)
