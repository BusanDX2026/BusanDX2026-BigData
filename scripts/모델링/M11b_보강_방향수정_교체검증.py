# -*- coding: utf-8 -*-
"""
M11b. M11 후속 — 방향 수정 · '추가' 대신 '교체' 검증

■ M11 결과
  흐름경로 −1.0% / 다중반경TPI −0.6% / 지하차도 −13.2% (위약 5개 −11.7%)
  → 전부 기각 방향. 다만 두 가지를 확인하고 확정한다.

  (1) flowpath_len_m 의 방향을 내가 반대로 가정했다.
      "경로 길수록 배수 지연 → 위험↑" 으로 놨는데 데이터는 정반대
      (침수 격자 0.179 vs 비침수 0.430). 수계에 가까울수록 잠긴다는 뜻이다.
      방향 자체가 틀렸으니 수정 후 재측정해야 공정하다.

  (2) 피처를 '더하면' 차원이 늘어 공간CV에서 불리하다. 기존 피처를 '갈아끼우는'
      교체 실험이 더 공정하다 — flowpath ↔ dist_stream, tpi1500 ↔ tpi(500m).

- 입력: 04_모델/features_v5.parquet
- 출력: _리포트/M11b_방향수정_교체.md
"""
import sys, io, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"
MOD = GG / "04_모델"; REP = GG / "_리포트"
RS = 42
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# M11b. 방향 수정 · 교체 검증")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

df = pd.read_parquet(MOD / "features_v5.parquet")
cfg = json.load(open(MOD / "M3_최종설정.json", encoding="utf-8"))
LC = ["imperv_ratio", "agri_ratio", "paddy_ratio", "forest_ratio", "water_ratio", "road_ratio"]
BASE16 = cfg["features"] + [c + "_s" for c in LC] + ["mh_no_dredge_ratio_s"]

# 방향 수정: 흐름경로는 짧을수록 위험 (수계 근접) → 반전
df["flowpath_inv_s"] = 1 - df["flowpath_len_m_s"]
a = df.loc[df.trace_flag == 1, "flowpath_inv_s"].mean()
z = df.loc[df.trace_flag == 0, "flowpath_inv_s"].mean()
P(f"방향 수정 후 flowpath_inv_s: 침수O {a:.3f} | 침수X {z:.3f} | {a-z:+.3f}  "
  f"({'OK' if a > z else '역전'})\n")

# 상관 — 교체 후보끼리 얼마나 겹치나
P("## 1. 기존 피처와의 상관 (겹치면 추가해도 정보가 없다)")
P("| 신규 | 기존 | Pearson r |")
P("|---|---|--:|")
for new, old in [("flowpath_inv_s", "dist_stream_m_s"), ("tpi_1500_s", "tpi_s"),
                 ("tpi_300_s", "tpi_s"), ("underpass_dist_m_s", "imperv_ratio_s")]:
    P(f"| {new} | {old} | {np.corrcoef(df[new], df[old])[0,1]:+.3f} |")

y = df.trace_flag.values; groups = df.sgg_cd.values
gkf = list(GroupKFold(n_splits=5).split(df, y, groups))
tie = np.random.RandomState(RS).rand(len(y))

def mk():
    return XGBClassifier(**cfg["params"], scale_pos_weight=cfg["scale_pos_weight"],
                         eval_metric="aucpr", tree_method="hist", reg_lambda=1.0,
                         n_jobs=-1, random_state=RS, verbosity=0)

def ev(feats):
    X = df[feats].values; oof = np.zeros(len(y))
    for tr, te in gkf:
        m = clone(mk()); m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    s = oof + tie * max(np.ptp(oof), 1e-9) * 1e-9
    order = np.argsort(-s); k = int(len(y)*0.10); tp = int(y[order[:k]].sum())
    return average_precision_score(y, oof), roc_auc_score(y, oof), tp/y.sum(), tp/k

base = ev(BASE16)
P(f"\n## 2. 기준 M9 (16피처): PR-AUC **{base[0]:.4f}** · Top10% {base[2]:.1%}\n")

def swap(feats, old, new):
    f = list(feats); f[f.index(old)] = new; return f

CASES = [
    ("＋흐름경로(방향수정)",        BASE16 + ["flowpath_inv_s"]),
    ("흐름경로 ↔ 수계직선거리 교체", swap(BASE16, "dist_stream_m_s", "flowpath_inv_s")),
    ("tpi1500 ↔ tpi(500m) 교체",    swap(BASE16, "tpi_s", "tpi_1500_s")),
    ("＋tpi1500 만",                BASE16 + ["tpi_1500_s"]),
    ("＋지하차도 거리 1개만",        BASE16 + ["underpass_dist_m_s"]),
]
P("| 구성 | 피처수 | PR-AUC | 변화 | ROC-AUC | Top10% 포착 | 정밀도 |")
P("|---|--:|--:|--:|--:|--:|--:|")
P(f"| 기준 M9 | 16 | {base[0]:.4f} | — | {base[1]:.4f} | {base[2]:.1%} | {base[3]:.1%} |")
best = ("기준 M9", base[0])
for nm, f in CASES:
    r = ev(f)
    P(f"| {nm} | {len(f)} | {r[0]:.4f} | {(r[0]-base[0])/base[0]:+.1%} | "
      f"{r[1]:.4f} | {r[2]:.1%} | {r[3]:.1%} |")
    if r[0] > best[1]: best = (nm, r[0])

P(f"\n**최고: {best[0]} (PR-AUC {best[1]:.4f})**")
REP.mkdir(parents=True, exist_ok=True)
(REP / "M11b_방향수정_교체.md").write_text("\n".join(log), encoding="utf-8")
P("\n→ M11b_방향수정_교체.md 저장")
