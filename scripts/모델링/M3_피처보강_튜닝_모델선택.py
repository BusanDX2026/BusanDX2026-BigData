# -*- coding: utf-8 -*-
"""
M3. 물리 피처 보강(흐름누적·TWI) → 하이퍼파라미터 튜닝 → 최종 모델 선택

왜 흐름누적/TWI를 추가하는가
  M2에서 HAZARD 모델 PR-AUC 0.173에 그쳤다. 현재 지형 피처(표고·경사·TPI)는 '국지적 형태'만 보고
  '물이 어디로 모여드는가'(상류 집수면적)를 못 본다. 부산 침수의 60%가 하천범람 밖 내수침수인데
  (전처리 S5b), 내수는 본질적으로 집수-배수 문제다. 도시침수지도가 강서구만 있어(한계 L-2)
  이 물리량을 DEM에서 직접 계산해 채운다.
    - flow_acc : D8 흐름누적 (상류에서 이 셀로 모이는 셀 수)
    - twi      : ln(집수면적 / tan(경사)) 지형습윤지수 — 물이 고이는 경향의 표준 지표
    - hand     : 최근접 수계 대비 상대고도(간이) — 저지대 중 '물길에 가까운' 곳 식별

- 입력: 00_정합_5186/DEM_30m_부산_5186.tif, 04_모델/features.parquet
- 출력: 04_모델/features_v2.parquet, 04_모델/M3_튜닝결과.csv
- 규약: random_state=42
"""
import sys, io, json, time, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"
MOD = GG / "04_모델"
REP = GG / "_리포트"
RS = 42
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# M3. 피처 보강 · 튜닝 · 모델 선택")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

df = pd.read_parquet(MOD / "features.parquet")
GROUPS = json.load(open(MOD / "feature_groups.json", encoding="utf-8"))

# ===============================================================
# 1. D8 흐름누적 · TWI · HAND 계산
# ===============================================================
P("## 1. DEM 수문 파생 (D8 흐름누적 · TWI · HAND)")
t0 = time.perf_counter()
with rasterio.open(GG / "00_정합_5186" / "DEM_30m_부산_5186.tif") as ds:
    dem = ds.read(1).astype("float64"); nd = ds.nodata
    src_t, src_crs = ds.transform, ds.crs; px = src_t.a
dem[dem == nd] = np.nan
H, W = dem.shape
valid = np.isfinite(dem)
P(f"- DEM {H}x{W}, 유효 {valid.sum():,}셀, 픽셀 {px:.0f}m")

# 미세 함몰 제거: 8이웃 최소값보다 낮으면 살짝 올림 (완전한 sink fill 대신 경량 처리)
# 2026-09-04 코드리뷰 #1 수정: 기존 인라인 D8은 하강 조건(drop>0)이 없어
#   5.5%(47,852셀)가 오르막으로 배수됐고 싱크를 42개로 오보했다. 공용 모듈로 교체.
#   (함몰메움 → drop>0 D8 → 누적 → 수계망 → HAND/TWI/수계거리)
from _hydro import derive_all
hyd = derive_all(dem, valid, px, stream_km2=1.0, verbose_fn=P)
flow_acc = hyd["flow_acc"]
twi = hyd["twi"]
dist_to_stream = np.where(valid, hyd["dist_stream"], np.nan)
P(f"- 수문 파생 완료 ({time.perf_counter()-t0:.1f}s), "
  f"평균 수계거리 {np.nanmean(dist_to_stream):.0f}m")

# ---- 100m 격자로 리샘플 (S5c와 동일 정렬) ----
CELL = 100
x_min = df.x_cen.min() - CELL/2; y_max = df.y_cen.max() + CELL/2
ncol = int(round((df.x_cen.max() + CELL/2 - x_min)/CELL))
nrow = int(round((y_max - (df.y_cen.min() - CELL/2))/CELL))
dst_t = from_origin(x_min, y_max, CELL, CELL)
def to_grid(arr, how=Resampling.average):
    out = np.full((nrow, ncol), np.nan, dtype="float32")
    reproject(arr.astype("float32"), out, src_transform=src_t, src_crs=src_crs,
              dst_transform=dst_t, dst_crs=src_crs, src_nodata=np.nan, dst_nodata=np.nan,
              resampling=how)
    return out
col = ((df.x_cen.values - x_min)/CELL).astype(int)
row = ((y_max - df.y_cen.values)/CELL).astype(int)
assert col.min() >= 0 and col.max() < ncol and row.min() >= 0 and row.max() < nrow

g_facc = to_grid(np.log1p(flow_acc), Resampling.max)      # 셀 내 최대 집수 (물길이 지나면 크게)
g_twi  = to_grid(twi, Resampling.average)
g_d2s  = to_grid(np.where(valid, dist_to_stream, np.nan), Resampling.average)
df["flow_acc_log"] = g_facc[row, col]
df["twi"] = g_twi[row, col]
df["dist_stream_m"] = g_d2s[row, col]
for c in ["flow_acc_log", "twi", "dist_stream_m"]:
    n_na = int(df[c].isna().sum())
    df[c] = df[c].fillna(df.groupby("sgg_cd")[c].transform("median")).fillna(df[c].median())
    if n_na: P(f"- `{c}` 결측 {n_na:,} → 자치구 중앙값")
P(f"- 격자 결합 완료 (총 {time.perf_counter()-t0:.1f}s)")

# 정규화 (S0 §6) — 위험↑=점수↑
NEW = {"flow_acc_log": False, "twi": False, "dist_stream_m": True}   # 수계 거리는 멀수록 안전 → 반전
for c, inv in NEW.items():
    v = df[c].astype(float)
    lo, hi = np.percentile(v, [5, 95]); hi = hi if hi > lo else lo + 1e-9
    s = np.clip((v - lo)/(hi - lo), 0, 1)
    df[c + "_s"] = 1 - s if inv else s
P("\n방향 검증 (침수흔적 격자에서 점수가 높아야 정상):")
for c in NEW:
    a = df.loc[df.trace_flag==1, c+"_s"].mean(); zz = df.loc[df.trace_flag==0, c+"_s"].mean()
    P(f"  {'OK ' if a>zz else '역전'} {c+'_s':<20} 침수O {a:.3f} | 침수X {zz:.3f} | {a-zz:+.3f}")

df.to_parquet(MOD / "features_v2.parquet", index=False)

# ===============================================================
# 2. 피처 보강 효과 측정
# ===============================================================
P("\n## 2. 피처 보강 효과 (자치구 GroupKFold5, XGBoost 고정설정)")
y = df.trace_flag.values; groups = df.sgg_cd.values
gkf = list(GroupKFold(n_splits=5).split(df, y, groups))
spw = (y == 0).sum() / y.sum()
HAZ0 = [c+"_s" for c in GROUPS["HAZARD"]]
HAZ1 = HAZ0 + ["flow_acc_log_s", "twi_s", "dist_stream_m_s"]

def oof_score(X, y, splits, model):
    oof = np.zeros(len(y))
    for tr, te in splits:
        m = clone(model); m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    return oof

xgb_base = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="aucpr",
                         tree_method="hist", n_jobs=-1, random_state=RS, verbosity=0)
res_aug = []
for feats, nm in [(HAZ0, "기존 HAZARD 6"), (HAZ1, "＋수문 3 (총 9)")]:
    oof = oof_score(df[feats].values, y, gkf, xgb_base)
    ap, auc = average_precision_score(y, oof), roc_auc_score(y, oof)
    res_aug.append((nm, len(feats), ap, auc))
    P(f"  {nm:<18} 피처{len(feats):>2} | PR-AUC {ap:.4f} | ROC {auc:.4f}")
gain = res_aug[1][2]/res_aug[0][2] - 1
P(f"\n→ 수문 피처 추가 효과: PR-AUC **{gain:+.1%}**")
P("  " + ("채택: 물리적 근거(집수-배수)와 성능 개선이 함께 확인됨." if gain > 0.02
          else "개선 미미 → 채택하되 과신 금물. 내수 침수의 대리변수로 해석만."))

# ===============================================================
# 3. 하이퍼파라미터 튜닝 (공간 CV 기준)
# ===============================================================
P("\n## 3. 하이퍼파라미터 튜닝 (자치구 GroupKFold5, PR-AUC 기준)")
P("  ※ 랜덤 CV로 튜닝하면 M2에서 확인한 +143~193% 과대평가를 그대로 최적화하게 되므로 반드시 공간 CV로 튜닝")
X1 = df[HAZ1].values
grid = [
    dict(max_depth=3, n_estimators=300, learning_rate=0.05, min_child_weight=5,  subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=4, n_estimators=400, learning_rate=0.05, min_child_weight=10, subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=5, n_estimators=400, learning_rate=0.05, min_child_weight=10, subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=6, n_estimators=600, learning_rate=0.03, min_child_weight=20, subsample=0.7, colsample_bytree=0.7),
    dict(max_depth=4, n_estimators=800, learning_rate=0.02, min_child_weight=20, subsample=0.7, colsample_bytree=0.7),
    dict(max_depth=3, n_estimators=600, learning_rate=0.03, min_child_weight=30, subsample=0.7, colsample_bytree=0.6),
]
rows = []
for i, g in enumerate(grid, 1):
    mdl = XGBClassifier(**g, scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist",
                        reg_lambda=1.0, n_jobs=-1, random_state=RS, verbosity=0)
    t0 = time.perf_counter(); oof = oof_score(X1, y, gkf, mdl); dt = time.perf_counter()-t0
    ap = average_precision_score(y, oof); auc = roc_auc_score(y, oof)
    rows.append(dict(cfg=i, **g, PR_AUC=ap, ROC_AUC=auc, fit_s=round(dt, 1)))
    P(f"  cfg{i} depth={g['max_depth']} n={g['n_estimators']} lr={g['learning_rate']} "
      f"mcw={g['min_child_weight']} → PR-AUC {ap:.4f} | ROC {auc:.4f} | {dt:.1f}s")
tune = pd.DataFrame(rows).sort_values("PR_AUC", ascending=False)
tune.to_csv(MOD / "M3_튜닝결과.csv", index=False, encoding="utf-8-sig")
best_cfg = {k: tune.iloc[0][k] for k in grid[0]}
for k in ["max_depth", "n_estimators", "min_child_weight"]:
    best_cfg[k] = int(best_cfg[k])
P(f"\n→ 최적 cfg{int(tune.iloc[0].cfg)}: {best_cfg}  (PR-AUC {tune.iloc[0].PR_AUC:.4f})")
P(f"  튜닝 전(cfg3 = M2 설정) 대비 {tune.iloc[0].PR_AUC/float(tune[tune.cfg==3].PR_AUC.iloc[0])-1:+.1%}")

# ===============================================================
# 4. 최종 모델 선택 — 성능 vs 해석가능성
# ===============================================================
P("\n## 4. 최종 모델 선택")
xgb_best = XGBClassifier(**best_cfg, scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist",
                         reg_lambda=1.0, n_jobs=-1, random_state=RS, verbosity=0)
lr = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RS)
cands = {"XGBoost(튜닝)": xgb_best, "LogisticRegression": lr}
sel = []
for nm, mdl in cands.items():
    t0 = time.perf_counter(); oof = oof_score(X1, y, gkf, mdl); dt = time.perf_counter()-t0
    k5 = int(len(y)*0.05); k10 = int(len(y)*0.10)
    rng = np.random.RandomState(RS).rand(len(y))
    o = oof + rng*np.ptp(oof)*1e-9
    t5 = y[np.argpartition(-o, k5-1)[:k5]].sum()/y.sum()
    t10 = y[np.argpartition(-o, k10-1)[:k10]].sum()/y.sum()
    sel.append(dict(model=nm, PR_AUC=average_precision_score(y, oof), ROC_AUC=roc_auc_score(y, oof),
                    Top5=t5, Top10=t10, fit_s=round(dt, 1)))
    np.save(MOD / f"oof_{'xgb' if 'XGB' in nm else 'lr'}.npy", oof)
sdf = pd.DataFrame(sel)
P("| 모델 | PR-AUC | ROC-AUC | Top5% | Top10% | fit(s) |")
P("|---|--:|--:|--:|--:|--:|")
for _, r in sdf.iterrows():
    P(f"| {r.model} | **{r.PR_AUC:.4f}** | {r.ROC_AUC:.4f} | {r.Top5:.1%} | {r.Top10:.1%} | {r.fit_s} |")
d_pr = sdf.PR_AUC.iloc[0]/sdf.PR_AUC.iloc[1] - 1
P(f"\n- XGBoost가 로지스틱 대비 PR-AUC {d_pr:+.1%}, 학습비용 {sdf.fit_s.iloc[0]/max(sdf.fit_s.iloc[1],0.01):.0f}배")
P("- **채택: XGBoost(튜닝)**. 근거 ① 지형-침수 관계가 비선형·임계적(표고 몇 m 이하에서 급증)이라 선형모델이 부적합"
  " ② 다중공선성에 강건(코드리뷰에서 확인된 지형 피처 간 r 0.7~0.9)"
  " ③ SHAP으로 격자별 기여도 분해가 가능해 '왜 이 구역인가'를 행정에 설명할 수 있음")
P("- 로지스틱은 **보조 지표**로 유지: 계수 부호로 물리적 타당성을 상시 점검(부호가 뒤집히면 데이터 이상 신호)")
json.dump({"features": HAZ1, "params": best_cfg, "scale_pos_weight": float(spw)},
          open(MOD / "M3_최종설정.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
P(f"\n- 산출: features_v2.parquet, M3_튜닝결과.csv, M3_최종설정.json, oof_*.npy")

(REP / "M3_피처보강_튜닝.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'M3_피처보강_튜닝.md'}", flush=True)
