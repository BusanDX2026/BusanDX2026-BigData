# -*- coding: utf-8 -*-
"""
M11. 피처 보강 3종 — 흐름경로 길이 · 다중반경 TPI · 지하차도

■ 배경
  D8 버그 수정(+25%) 이후 남은 개선 여지를 찾는다. 새 외부 데이터 없이
  **이미 가진 DEM·지하차도**에서 물리적으로 정당한 피처만 뽑는다.

■ 3종
  ① flowpath_len_m  수계까지 D8 흐름경로 길이
     - 당초 계획한 '배수 방류구(토구) 거리'는 저장소에 원천 데이터가 없다.
       물리적으로 같은 것을 재는 대체 지표. 직선거리(dist_stream_m)와 달리
       능선을 넘지 않고 실제 배수 경로를 따라간다.
  ② tpi_300 / tpi_1500  다중반경 지형위치지수
     - 기존 tpi 는 500m(5x5) 단일 창. SHAP 상위가 전부 '상대지형'이므로
       스케일을 분해하면 미소 함몰(300m)과 곡저 위치(1500m)를 나눠 볼 수 있다.
  ③ underpass_dist_m / underpass_n_500m  지하차도
     - 02_레이어별에 이미 계산돼 있으나 모델에 넣은 적이 없다.
     - 역인과 아님: 지하차도는 침수 때문에 만들지 않는다(도로 입체교차 때문).
       물리적으로는 강제된 국지 최저점 = 집수점.

■ 검증
  단계별 누적 + **위약 대조**(행정동 난수). M8 교훈 — 성능 상승만으로 채택하지 않고
  SHAP 방향이 물리적으로 타당한지 함께 본다.

- 입력: 04_모델/features_v4.parquet, 00_정합_5186/DEM_30m_부산_5186.tif,
        02_레이어별/지하차도_grid.parquet
- 출력: 04_모델/features_v5.parquet, _리포트/M11_피처보강.md
"""
import sys, io, json, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from scipy.ndimage import uniform_filter
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"
MOD = GG / "04_모델"; REP = GG / "_리포트"; LAY = GG / "02_레이어별"
RS = 42
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# M11. 피처 보강 3종 (흐름경로 · 다중반경 TPI · 지하차도)")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

df = pd.read_parquet(MOD / "features_v4.parquet")
cfg = json.load(open(MOD / "M3_최종설정.json", encoding="utf-8"))
LC = ["imperv_ratio", "agri_ratio", "paddy_ratio", "forest_ratio", "water_ratio", "road_ratio"]
BASE16 = cfg["features"] + [c + "_s" for c in LC] + ["mh_no_dredge_ratio_s"]
P(f"기준 모델: {len(BASE16)}피처 (M9) · 격자 {len(df):,} · 양성 {df.trace_flag.sum():,}\n")

# ===============================================================
# 1. DEM 파생 — 흐름경로 길이 · 다중반경 TPI
# ===============================================================
P("## 1. DEM 파생")
t0 = time.perf_counter()
with rasterio.open(GG / "00_정합_5186" / "DEM_30m_부산_5186.tif") as ds:
    dem = ds.read(1).astype("float64"); nd = ds.nodata
    src_t, src_crs = ds.transform, ds.crs; px = src_t.a
dem[dem == nd] = np.nan
valid = np.isfinite(dem)
P(f"- DEM {dem.shape[0]}x{dem.shape[1]}, 유효 {valid.sum():,}셀, 픽셀 {px:.0f}m")

from _hydro import derive_all, flowpath_to_stream
hyd = derive_all(dem, valid, px, stream_km2=1.0, verbose_fn=P)
fpl = flowpath_to_stream(hyd["filled"], hyd["best_nb"], hyd["stream"], valid, px)
n_nan = int((valid & ~np.isfinite(fpl)).sum())
P(f"- 흐름경로 길이: 미도달 {n_nan:,}셀 ({n_nan/valid.sum():.2%}) → NaN 유지")
P(f"  중앙값 {np.nanmedian(fpl):,.0f}m / 최대 {np.nanmax(fpl):,.0f}m "
  f"(직선거리 중앙값 {np.nanmedian(hyd['dist_stream']):,.0f}m)")
P(f"- DEM 파생 완료 ({time.perf_counter()-t0:.1f}s)")

# ---- 100m 격자 정렬 (S5c·M3 와 동일) ----
CELL = 100
x_min = df.x_cen.min() - CELL/2; y_max = df.y_cen.max() + CELL/2
ncol = int(round((df.x_cen.max() + CELL/2 - x_min)/CELL))
nrow = int(round((y_max - (df.y_cen.min() - CELL/2))/CELL))
dst_t = from_origin(x_min, y_max, CELL, CELL)

def to_grid(arr, how=Resampling.average):
    out = np.full((nrow, ncol), np.nan, dtype="float32")
    reproject(arr.astype("float32"), out, src_transform=src_t, src_crs=src_crs,
              dst_transform=dst_t, dst_crs=src_crs, src_nodata=np.nan,
              dst_nodata=np.nan, resampling=how)
    return out

col = ((df.x_cen.values - x_min)/CELL).astype(int)
row = ((y_max - df.y_cen.values)/CELL).astype(int)
assert col.min() >= 0 and col.max() < ncol and row.min() >= 0 and row.max() < nrow

r_fpl = to_grid(fpl)
r_elev = to_grid(np.where(valid, dem, np.nan))

# 다중반경 TPI — 100m 격자 평균표고 기준. WIN=3 → 300m, WIN=15 → 1500m
def tpi_at(r_mean, win):
    fin = np.isfinite(r_mean)
    neigh = uniform_filter(np.where(fin, r_mean, 0), size=win)
    cnt = uniform_filter(fin.astype("float32"), size=win)
    neigh = np.where(cnt > 0, neigh/np.maximum(cnt, 1e-6), np.nan)
    return (r_mean - neigh).astype("float32")

r_tpi300, r_tpi1500 = tpi_at(r_elev, 3), tpi_at(r_elev, 15)
df["flowpath_len_m"] = r_fpl[row, col]
df["tpi_300"] = r_tpi300[row, col]
df["tpi_1500"] = r_tpi1500[row, col]
P(f"- 다중반경 TPI: 300m(3x3) · 1500m(15x15) 산출 (기존 tpi = 500m)")

# ===============================================================
# 2. 지하차도 결합
# ===============================================================
P("\n## 2. 지하차도 결합")
up = pd.read_parquet(LAY / "지하차도_grid.parquet")[
    ["grid_id", "underpass_dist_m", "underpass_n_500m"]]
df = df.merge(up, on="grid_id", how="left")
P(f"- 지하차도 보유 격자(500m내 1개 이상): {int((df.underpass_n_500m > 0).sum()):,}")

# ===============================================================
# 3. 결측 처리 · 정규화 (S0 §6 — 위험↑=점수↑)
# ===============================================================
P("\n## 3. 결측 · 정규화")
NEW = {"flowpath_len_m": False,     # 경로 길수록 배수 지연 → 위험↑
       "tpi_300": True,             # TPI 낮을수록(주변보다 낮음) 위험↑ → 반전
       "tpi_1500": True,
       "underpass_dist_m": True,    # 가까울수록 위험↑ → 반전
       "underpass_n_500m": False}
for c in NEW:
    n_na = int(df[c].isna().sum())
    if n_na:
        df[c] = df[c].fillna(df.groupby("sgg_cd")[c].transform("median")).fillna(df[c].median())
        P(f"- `{c}` 결측 {n_na:,} → 자치구 중앙값")
for c, inv in NEW.items():
    v = df[c].astype(float)
    lo, hi = np.percentile(v, [5, 95]); hi = hi if hi > lo else lo + 1e-9
    s = np.clip((v - lo)/(hi - lo), 0, 1)
    df[c + "_s"] = 1 - s if inv else s

P("\n방향 검증 (침수흔적 격자에서 점수가 높아야 정상):")
for c in NEW:
    a = df.loc[df.trace_flag == 1, c+"_s"].mean(); z = df.loc[df.trace_flag == 0, c+"_s"].mean()
    P(f"  {'OK  ' if a > z else '역전'} {c+'_s':<22} 침수O {a:.3f} | 침수X {z:.3f} | {a-z:+.3f}")

df.to_parquet(MOD / "features_v5.parquet", index=False)

# ===============================================================
# 4. 누적 절제 + 위약 대조
# ===============================================================
P("\n## 4. 성능 (자치구 GroupKFold5 OOF, M3 튜닝설정 고정)")
y = df.trace_flag.values; groups = df.sgg_cd.values
gkf = list(GroupKFold(n_splits=5).split(df, y, groups))
spw = cfg["scale_pos_weight"]
rng = np.random.RandomState(RS)
tie = rng.rand(len(y))

def mk():
    return XGBClassifier(**cfg["params"], scale_pos_weight=spw, eval_metric="aucpr",
                         tree_method="hist", reg_lambda=1.0, n_jobs=-1,
                         random_state=RS, verbosity=0)

def evaluate(feats, frame=None):
    d = frame if frame is not None else df
    X = d[feats].values
    oof = np.zeros(len(y))
    for tr, te in gkf:
        m = clone(mk()); m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    s = oof + tie * max(np.ptp(oof), 1e-9) * 1e-9
    order = np.argsort(-s); k = int(len(y) * 0.10)
    tp = int(y[order[:k]].sum())
    return dict(pr=average_precision_score(y, oof), roc=roc_auc_score(y, oof),
                rec10=tp/y.sum(), prec10=tp/k, oof=oof)

FP1 = ["flowpath_len_m_s"]
TPI2 = ["tpi_300_s", "tpi_1500_s"]
UP2 = ["underpass_dist_m_s", "underpass_n_500m_s"]
STAGES = [("기준 M9 (16피처)", BASE16),
          ("＋흐름경로 길이 1", BASE16 + FP1),
          ("＋다중반경 TPI 2", BASE16 + FP1 + TPI2),
          ("＋지하차도 2", BASE16 + FP1 + TPI2 + UP2)]

P("| 단계 | 피처수 | PR-AUC | 변화 | ROC-AUC | Top10% 포착 | 정밀도 |")
P("|---|--:|--:|--:|--:|--:|--:|")
res, prev, base_pr = {}, None, None
for nm, feats in STAGES:
    r = evaluate(feats); res[nm] = r
    if base_pr is None: base_pr = r["pr"]
    ch = "—" if prev is None else f"{(r['pr']-prev)/prev:+.1%}"
    P(f"| {nm} | {len(feats)} | {r['pr']:.4f} | {ch} | {r['roc']:.4f} | "
      f"{r['rec10']:.1%} | {r['prec10']:.1%} |")
    prev = r["pr"]
full = STAGES[-1][1]
P(f"\n누적 변화: {base_pr:.4f} → {res[STAGES[-1][0]]['pr']:.4f} "
  f"(**{(res[STAGES[-1][0]]['pr']-base_pr)/base_pr:+.1%}**)")

# ---- 개별 기여 (기준 + 1군씩) ----
P("\n### 개별 기여 (기준16 ＋ 해당 군만)")
P("| 추가 군 | PR-AUC | 변화 |")
P("|---|--:|--:|")
for nm, add in [("흐름경로 1", FP1), ("다중반경 TPI 2", TPI2), ("지하차도 2", UP2)]:
    r = evaluate(BASE16 + add)
    P(f"| {nm} | {r['pr']:.4f} | {(r['pr']-base_pr)/base_pr:+.1%} |")

# ---- 위약 대조 (M8 교훈) ----
P("\n### 위약 대조 — 행정동 난수 5개 (M8 절차)")
dp = df.copy()
adm = dp.adm_cd.astype(str)
for i in range(5):
    mp = {a: rng.rand() for a in adm.unique()}
    dp[f"placebo{i}_s"] = adm.map(mp).values
rp = evaluate(BASE16 + [f"placebo{i}_s" for i in range(5)], frame=dp)
P(f"| 위약 5개 | {rp['pr']:.4f} | {(rp['pr']-base_pr)/base_pr:+.1%} |")
P("\n→ 위약이 확실히 떨어져야 채택 절차가 잡음에 속지 않는다는 뜻이다.")

# ---- SHAP 방향 ----
P("\n## 5. SHAP 방향 검증 (신규 5피처)")
import shap
Xf = df[full].values
m = clone(mk()); m.fit(Xf, y)
sv = shap.TreeExplainer(m).shap_values(Xf)
imp = np.abs(sv).mean(0); share = imp/imp.sum()
P("| 피처 | SHAP 기여율 | 방향(값↑ 시 위험) | 판정 |")
P("|---|--:|---|---|")
for c in FP1 + TPI2 + UP2:
    i = full.index(c)
    d = np.corrcoef(Xf[:, i], sv[:, i])[0, 1]
    P(f"| {c} | {share[i]:.1%} | {'+' if d > 0 else '−'} | "
      f"{'물리 타당' if d > 0 else '**역전 — 재검토**'} |")
P("\n※ 정규화에서 이미 위험↑=점수↑ 로 방향을 맞췄으므로 SHAP 방향도 `+` 여야 정상.")

REP.mkdir(parents=True, exist_ok=True)
(REP / "M11_피처보강.md").write_text("\n".join(log), encoding="utf-8")
json.dump({"features_full": full, "new": FP1+TPI2+UP2,
           "pr_base": float(base_pr), "pr_full": float(res[STAGES[-1][0]]["pr"])},
          open(MOD / "M11_피처보강.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
P(f"\n→ features_v5.parquet · M11_피처보강.md 저장")
