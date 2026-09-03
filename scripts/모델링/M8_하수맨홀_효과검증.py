# -*- coding: utf-8 -*-
"""
M8. 하수맨홀 상태지표(행정동 단위 비율) 추가 효과 검증

배경
  내수침수의 근본 원인 데이터인 하수관로는 부산이 미공개(지하시설물 보안).
  대신 하수맨홀 100,646건이 공개돼 있으나 **좌표가 없고 최세밀 공간정보가 행정동**이다.

설계 원칙 — 절대 개수는 쓰지 않는다
  등록구별 맨홀 수가 최대/최소 58.5배(기장 13,516 vs 중구 1,253)로 벌어진다.
  중구는 부산 최고밀집 도심인데 등록이 가장 적어, 실제 시설 차이가 아니라
  **구별 DB 등록 성실도 차이**로 판단된다 — 침수흔적도와 같은 종류의 편향(이슈 #13).
  → 밀도(개수) 지표는 배제하고 **'등록된 맨홀 중 어떤 상태인가'** 비율 지표만 사용한다.

지표 5종 (전부 비율/중앙값 → 등록량에 불변)
  mh_combined_ratio  합류+집수+차집 비율   — 합류식은 폭우 시 우수·오수 동시 유입 → 역류 위험
  mh_defect_ratio    불량+준설필요 비율     — 하수도 취약 직접 지표
  mh_no_dredge_ratio 준설이력 없음 비율     — 준설 미실시 = 막힘 위험 (전체의 62.6%)
  mh_age             2026 − 설치연도 중앙값 — 노후도
  mh_storm_ratio     우수맨홀 비율          — 우수 배제계통 비중 (방향은 데이터로 확인)

한계 (사전 명시)
  행정동 단위이므로 같은 행정동 내 모든 격자가 동일값 = 격자 모델엔 '행정동 상수'.
  자치구 GroupKFold 에서는 미지의 자치구로 일반화해야 하므로 기여가 제한적일 수 있다.

- 입력: raw/14_하수맨홀_부산/하수맨홀_부산_20251216.csv, 04_모델/features_v3.parquet
- 출력: 02_레이어별/하수맨홀_행정동.parquet, 04_모델/features_v4.parquet, _리포트/M8_하수맨홀효과.md
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
MOD = GG / "04_모델"; LYR = GG / "02_레이어별"; REP = GG / "_리포트"
SRC = ROOT / "공공데이터" / "raw" / "14_하수맨홀_부산" / "하수맨홀_부산_20251216.csv"
RS = 42
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# M8. 하수맨홀 상태지표 효과 검증")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

mh = pd.read_csv(SRC, encoding="cp949", dtype=str)
P(f"하수맨홀 원본 {len(mh):,}건")
mh = mh[mh.행정동.notna() & (mh.등록구 != "기타")].copy()
P(f"유효(행정동·등록구 있음) {len(mh):,}건")

COMB = {"합류", "집수", "차집시설"}
DEFECT = {"불량", "준설필요", "불량/준설필요", "폐쇄"}
mh["_comb"] = mh.우오수구분.isin(COMB)
mh["_def"] = mh.불량여부.isin(DEFECT)
mh["_nodredge"] = mh.최종준설일자.isna()
mh["_storm"] = mh.우오수구분.eq("우수")
mh["_yr"] = pd.to_datetime(mh.설치일자, errors="coerce").dt.year

agg = mh.groupby(["등록구", "행정동"]).agg(
    n=("우오수구분", "size"),
    mh_combined_ratio=("_comb", "mean"),
    mh_defect_ratio=("_def", "mean"),
    mh_no_dredge_ratio=("_nodredge", "mean"),
    mh_storm_ratio=("_storm", "mean"),
    _yr_med=("_yr", "median"),
).reset_index()
agg["mh_age"] = 2026 - agg._yr_med
agg = agg.rename(columns={"등록구": "sgg_nm", "행정동": "adm_nm"}).drop(columns="_yr_med")
agg.to_parquet(LYR / "하수맨홀_행정동.parquet", index=False)
P(f"행정동 집계 {len(agg)}개")

df = pd.read_parquet(MOD / "features_v3.parquet")
n0 = len(df)
df = df.merge(agg.drop(columns="n"), on=["sgg_nm", "adm_nm"], how="left")
assert len(df) == n0
MH = ["mh_combined_ratio", "mh_defect_ratio", "mh_no_dredge_ratio", "mh_storm_ratio", "mh_age"]
miss = df[MH].isna().mean()
P(f"격자 매칭 결측률: {dict(miss.round(4))}")
for c in MH:
    df[c] = df[c].fillna(df[c].median())
# 정규화 [0,1] (mh_age 만 스케일 필요)
for c in MH:
    v = df[c].astype(float)
    lo, hi = np.percentile(v, [5, 95]); hi = hi if hi > lo else lo + 1e-9
    df[c + "_s"] = np.clip((v - lo) / (hi - lo), 0, 1)
df.to_parquet(MOD / "features_v4.parquet", index=False)

y = df.trace_flag.values; groups = df.sgg_cd.values
P(f"\n## 방향 검증 (침수흔적 격자에서 높아야 위험 방향)")
for c in MH:
    a = df.loc[y == 1, c].mean(); z = df.loc[y == 0, c].mean()
    P(f"  {'OK ' if a > z else '역전'} {c:<20} 침수O {a:.4f} | 침수X {z:.4f} | {a-z:+.4f}")

cfg = json.load(open(MOD / "M3_최종설정.json", encoding="utf-8"))
NEW6 = ["imperv_ratio", "agri_ratio", "paddy_ratio", "forest_ratio", "water_ratio", "road_ratio"]
BASE = cfg["features"] + [c + "_s" for c in NEW6]
PLUS = BASE + [c + "_s" for c in MH]
gkf = list(GroupKFold(n_splits=5).split(df, y, groups))

def mk():
    return XGBClassifier(**cfg["params"], scale_pos_weight=cfg["scale_pos_weight"],
                         eval_metric="aucpr", tree_method="hist", reg_lambda=1.0,
                         n_jobs=-1, random_state=RS, verbosity=0)

def oof_of(X):
    o = np.zeros(len(y))
    for tr, te in gkf:
        m = clone(mk()); m.fit(X[tr], y[tr]); o[te] = m.predict_proba(X[te])[:, 1]
    return o

def topn(yy, s, pct):
    k = max(1, int(len(s) * pct))
    o = s + np.random.RandomState(RS).rand(len(s)) * max(np.ptp(s), 1e-9) * 1e-9
    return yy[np.argpartition(-o, k - 1)[:k]].sum() / max(yy.sum(), 1)

P(f"\n## 성능 비교 (자치구 GroupKFold5)")
res = {}
for feats, nm in [(BASE, "기존 15피처"), (PLUS, "＋하수맨홀 5 (20)")]:
    o = oof_of(df[feats].values)
    res[nm] = dict(oof=o, ap=average_precision_score(y, o), auc=roc_auc_score(y, o),
                   t5=topn(y, o, .05), t10=topn(y, o, .10))
    r = res[nm]
    P(f"  {nm:<20} PR-AUC {r['ap']:.4f} | ROC {r['auc']:.4f} | Top5% {r['t5']:.1%} | Top10% {r['t10']:.1%}")
a, b = res["기존 15피처"], res["＋하수맨홀 5 (20)"]
P(f"\n→ PR-AUC **{(b['ap']/a['ap']-1):+.1%}** | Top10% {a['t10']:.1%} → **{b['t10']:.1%}** ({b['t10']-a['t10']:+.1%}p)")

P("\n## 자치구 내 상대순위")
rows = []
for gu in df.sgg_nm.unique():
    m = (df.sgg_nm == gu).values
    if y[m].sum() < 10: continue
    rows.append((gu, spearmanr(a["oof"][m], y[m]).statistic, spearmanr(b["oof"][m], y[m]).statistic))
gr = pd.DataFrame(rows, columns=["자치구", "before", "after"])
gr["delta"] = gr.after - gr.before
P(f"- ρ 중앙값 {gr.before.median():+.3f} → **{gr.after.median():+.3f}**")
P(f"- 개선된 자치구 {(gr.delta>0).sum()}/{len(gr)}")
for _, r in gr.sort_values("delta", ascending=False).head(4).iterrows():
    P(f"    {r.자치구} {r.before:+.3f} → {r.after:+.3f} ({r.delta:+.3f})")

P("\n## SHAP 기여 (하수맨홀 지표가 실제로 쓰이는가)")
mdl = mk().fit(df[PLUS].values, y)
samp = df.sample(n=min(20000, len(df)), random_state=RS)
sv = shap.TreeExplainer(mdl).shap_values(samp[PLUS].values)
imp = pd.DataFrame({"feature": PLUS, "shap": np.abs(sv).mean(0)})
imp["dir"] = [np.corrcoef(samp[f].values, sv[:, i])[0, 1] for i, f in enumerate(PLUS)]
imp["share"] = imp.shap / imp.shap.sum()
imp = imp.sort_values("shap", ascending=False)
imp.to_csv(MOD / "M8_shap_importance.csv", index=False, encoding="utf-8-sig")
mh_share = imp[imp.feature.str.startswith("mh_")].share.sum()
P(f"- 하수맨홀 5지표 합산 기여율 **{mh_share:.1%}**")
for _, r in imp[imp.feature.str.startswith("mh_")].iterrows():
    P(f"    {r.feature:<24} {r.share:5.1%}  {'↑' if r['dir']>0 else '↓'} ({r['dir']:+.2f})")
P("\n상위 8개 피처:")
for _, r in imp.head(8).iterrows():
    P(f"    {r.feature:<24} {r.share:5.1%}  {'↑' if r['dir']>0 else '↓'} ({r['dir']:+.2f})")

P("\n## 판정")
gain = b["ap"] / a["ap"] - 1
P(f"- PR-AUC 변화 {gain:+.1%} → " + ("**채택**" if gain > 0.02 else
   "**기여 미미** — 행정동 단위 상수라는 한계가 확인됨. 격자 모델엔 넣지 않고 행정동 우선순위 산출에만 사용 권장"))
P(f"- 산출: 02_레이어별/하수맨홀_행정동.parquet, features_v4.parquet, M8_shap_importance.csv")

(REP / "M8_하수맨홀효과.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'M8_하수맨홀효과.md'}", flush=True)
