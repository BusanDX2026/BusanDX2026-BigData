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

P("\n## 3. [V1] 사건 일반화 (라벨 민감도)")
# ⚠ 이것은 '해당 격자를 학습에서 제거'하는 홀드아웃이 아니다. 격자는 학습셋에 남기고
#   라벨만 1→0 으로 바꾼 뒤(y0), 그 격자를 다시 양성으로 놓고 평가한다.
#   보고서·표에 "학습 제외"라고 쓰면 안 된다 (이슈 #43).
# ⚠ 이 수치는 V2·V3·V5 가 읽어 쓴다. 산출 스크립트에 하드코딩하지 말 것
#   → 아래에서 04_모델/M9_검증.json 으로 내보낸다.
HOLDOUTS = []
for _yr, _nm in [(2014, "2014-08-25"), (2020, "2020 대형호우"), (2011, "2011-07-27")]:
    _h = ((df.trace_flag == 1) & (df.trace_last_year == _yr)).values
    if _h.sum() == 0:
        continue
    _y0 = np.where(_h, 0, y)
    _spw = (_y0 == 0).sum() / max(_y0.sum(), 1)
    _o = oof_of(X, _y0, list(GroupKFold(n_splits=5).split(X, _y0, groups)), _spw)
    _m = _h | (y == 0)
    _ap = average_precision_score(y[_m], _o[_m]); _b = y[_m].mean()
    _k = int(_m.sum() * .10)
    _s = _o[_m] + tie[_m] * max(np.ptp(_o[_m]), 1e-9) * 1e-9
    _t = y[_m][np.argpartition(-_s, _k - 1)[:_k]].sum() / max(y[_m].sum(), 1)
    HOLDOUTS.append(dict(event=_nm, n=int(_h.sum()), ap=float(_ap), base=float(_b),
                         lift=float(_ap / _b), top10=float(_t)))
P("| 사건 | 해당 격자 | AP | 양성률 | 리프트 | Top10% 포착 |")
P("|---|--:|--:|--:|--:|--:|")
for _r in HOLDOUTS:
    P(f"| {_r['event']} | {_r['n']:,} | {_r['ap']:.4f} | {_r['base']:.4f} | "
      f"**{_r['lift']:.1f}배** | {_r['top10']:.1%} |")
P("\n※ 해당 격자를 제거한 것이 아니라 **라벨을 1→0 으로 바꾼 민감도 분석**이다.")

P("\n## 3-1. 행정 재해위험지구 대비 (동일 격자 수 · OOF 기준)")
HAZD = []
_bl = pd.read_parquet(GG / "03_마스터" / "baseline_grid.parquet")
_d = df[["grid_id"]].merge(_bl, on="grid_id", how="left")
for _c in ["hazdist_flag", "hazdist_flood", "hazdist_flood_active"]:
    if _c not in _d.columns:
        continue
    _mm = _d[_c].fillna(0).astype(int).values == 1
    _K = int(_mm.sum())
    if _K == 0:
        continue
    _base = float(y[_mm].mean())
    _top = float(y[order[:_K]].mean())
    HAZD.append(dict(flag=_c, k=_K, base=_base, model=_top,
                     ratio=(_top / _base) if _base else None))
P("| 지정 기준 | 격자 | 지정구역 정밀도 | M9 상위 동수 격자 | 배수 |")
P("|---|--:|--:|--:|--:|")
for _r in HAZD:
    P(f"| {_r['flag']} | {_r['k']:,} | {_r['base']:.1%} | {_r['model']:.1%} | **{_r['ratio']:.2f}배** |")
_out = _d.hazdist_flag.fillna(0).astype(int).values == 1
OUTSIDE = float(1 - y[_out].sum() / max(int(y.sum()), 1))
P(f"\n- 침수흔적 격자 중 재해위험지구(hazdist_flag) **밖** 비율: **{OUTSIDE:.1%}**")

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
json.dump({"pr_auc": float(average_precision_score(y, oof)), "base_rate": float(y.mean()),
           "holdouts": HOLDOUTS, "hazdist": HAZD, "outside_ratio": OUTSIDE},
          open(MOD / "M9_검증.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
P(f"\n- 산출: hazard_score.parquet(갱신), M9_최종설정.json, M9_검증.json, M9_shap.csv")

(REP / "M9_최종모델.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'M9_최종모델.md'}", flush=True)
