# -*- coding: utf-8 -*-
"""
MS1. HAND(최근접 배수로 상대고도) 계산 + 물리 침수 감수성 지수(S_phys)

■ 왜 주제를 바꾸는가
  기존 M9(ML)는 침수흔적을 타깃으로 학습한다. 침수흔적은 자치구 신고편향(기장 886 vs 부산진 22)이
  크고 56%가 2014 단일사건이다. "강수량이 커지면 어느 구역을 선제대응해야 하나"로 주제를 바꾸면
  강우강도가 축(axis)이 되는데, 이때 필요한 것은 '강우에 물리적으로 올바르게 반응하는' 위험도다.
  M9의 rain 피처는 공간지문(부산 강우 std 5.4mm)이라 전역 perturbation 시 비단조(강우↑ → 위험격자↓).
  → 침수흔적 타깃을 쓰지 않는 '물리 감수성 지수'를 DEM+토지피복으로 직접 만든다.

■ S_phys 구성지표 (모두 지형/피복 기반, 인구·펌프·침수흔적 미사용)
  - HAND      : 격자 최저표고 − 최근접 배수로 표고 (수직 여유고). 낮을수록 위험. ★핵심
  - TWI       : ln(집수면적/tanβ). 높을수록 물이 고임
  - flow_acc  : D8 흐름누적(log). 높을수록 상류 유입 많음
  - slope     : 경사(도). 낮을수록 배수 안 됨
  - tpi       : 국지 상대표고. 낮을수록 주변보다 저지대
  - imperv    : 불투수율(토지피복). 높을수록 유출 급증 → 낮은 강우에도 침수
  - lowland3  : ≤3m 저지대 비율

■ 출력
  - 04_모델/features_scenario.parquet  (grid_id + hand_m + s_phys + 성분점수)
  - _리포트/MS1_HAND_물리취약성.md
"""
import sys, io, json, time, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"
MOD = GG / "04_모델"
REP = GG / "_리포트"
RS = 42
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# MS1. HAND 계산 + 물리 침수 감수성 지수(S_phys)")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

df = pd.read_parquet(MOD / "features_v4.parquet")
P(f"격자: {len(df):,}행")

# ===============================================================
# 1. DEM → D8 → 흐름누적 → 수계망 → HAND
# ===============================================================
P("## 1. HAND (Height Above Nearest Drainage) 계산")
t0 = time.perf_counter()
with rasterio.open(GG / "00_정합_5186" / "DEM_30m_부산_5186.tif") as ds:
    dem = ds.read(1).astype("float64"); nd = ds.nodata
    src_t, src_crs = ds.transform, ds.crs; px = src_t.a
dem[dem == nd] = np.nan
H, W = dem.shape
valid = np.isfinite(dem)
P(f"- DEM {H}x{W}, 유효 {valid.sum():,}셀, 픽셀 {px:.0f}m")

# 2026-09-04 코드리뷰 #1·#2 수정:
#   (구) D8에 하강 조건 없음 → 5.5%(47,852셀)가 오르막 배수, 싱크 42개로 오보
#   (구) HAND 미해결 시 자기표고 폴백 → hand=0(최대취약), 격자의 31.5%가 0m
#   (신) 함몰메움 → drop>0 D8 → 누적 → 수계망 → HAND(미해결 NaN). 공용 모듈 _hydro.
THR_KM2 = 1.0
from _hydro import derive_all
hyd = derive_all(dem, valid, px, stream_km2=THR_KM2, verbose_fn=P)
hand_2d = hyd["hand"].astype("float32")
vv = hyd["hand"][np.isfinite(hyd["hand"])]
P(f"- 완료 ({time.perf_counter()-t0:.1f}s). HAND 분포(m) p10/p50/p90 = "
  f"{np.percentile(vv,10):.1f} / {np.percentile(vv,50):.1f} / {np.percentile(vv,90):.1f}")

# ===============================================================
# 2. 100m 격자로 리샘플 (S5c / M3 와 동일 정렬)
# ===============================================================
CELL = 100
x_min = df.x_cen.min() - CELL/2; y_max = df.y_cen.max() + CELL/2
ncol = int(round((df.x_cen.max() + CELL/2 - x_min)/CELL))
nrow = int(round((y_max - (df.y_cen.min() - CELL/2))/CELL))
dst_t = from_origin(x_min, y_max, CELL, CELL)
def to_grid(arr, how):
    out = np.full((nrow, ncol), np.nan, dtype="float32")
    reproject(arr.astype("float32"), out, src_transform=src_t, src_crs=src_crs,
              dst_transform=dst_t, dst_crs=src_crs, src_nodata=np.nan, dst_nodata=np.nan,
              resampling=how)
    return out
col = ((df.x_cen.values - x_min)/CELL).astype(int)
row = ((y_max - df.y_cen.values)/CELL).astype(int)
assert col.min() >= 0 and col.max() < ncol and row.min() >= 0 and row.max() < nrow

g_hand = to_grid(hand_2d, Resampling.min)      # 격자 내 최저 HAND (가장 취약한 지점 기준)
df["hand_m"] = g_hand[row, col]
n_na = int(df.hand_m.isna().sum())
df["hand_m"] = df["hand_m"].fillna(df.groupby("sgg_cd")["hand_m"].transform("median")).fillna(df["hand_m"].median())
if n_na:
    P(f"- HAND 결측 {n_na:,} → 자치구 중앙값 대체")
P(f"- 격자 HAND(m): p10/p50/p90 = {df.hand_m.quantile([.1,.5,.9]).round(1).tolist()}")

# ===============================================================
# 3. 물리 감수성 지수 S_phys (침수흔적 타깃 미사용)
# ===============================================================
P("\n## 3. 물리 감수성 지수 S_phys")
# 성분: (지표, 반전여부=위험이 낮은값일때 True)  robust 5~95 정규화
# lowland3_ratio 제외 (코드리뷰 #9): 방향 검증에서 침수O 0.104 < 침수X 0.181 로 **역전**.
#   원인은 강서 삼각주 — 절대표고 ≤3m 저지대가 광대하나 농경지라 침수기록이 없음(이슈 #5).
#   상대지형(TPI·HAND)이 이미 저지대성을 담고 있으므로 제거하고 가중을 비례 재배분.
COMP = {
    "hand_m":          (True,  0.30),   # 낮을수록 위험 ★
    "twi":             (False, 0.21),
    "flow_acc_log":    (False, 0.15),
    "slope_mean":      (True,  0.13),
    "tpi":             (True,  0.13),
    "imperv_ratio":    (False, 0.08),
}
assert abs(sum(w for _, w in COMP.values()) - 1.0) < 1e-9
parts = {}
for c, (inv, w) in COMP.items():
    v = df[c].astype(float)
    lo, hi = np.nanpercentile(v, [5, 95]); hi = hi if hi > lo else lo + 1e-9
    s = np.clip((v - lo) / (hi - lo), 0, 1)
    s = 1 - s if inv else s
    parts[c + "_p"] = s
    df[c + "_p"] = s
df["s_phys"] = sum(parts[c + "_p"] * w for c, (_, w) in COMP.items())
df["s_phys_pct"] = df["s_phys"].rank(pct=True)
P("  가중치: " + ", ".join(f"{c}={w:.2f}" for c, (_, w) in COMP.items()))
P(f"  s_phys 분포 p10/p50/p90 = {df.s_phys.quantile([.1,.5,.9]).round(3).tolist()}")

# 방향 점검 — 침수흔적 격자에서 각 성분점수가 높아야 정상 (검증 목적, 학습 아님)
P("\n  성분 방향 점검 (침수O vs 침수X 평균):")
for c in COMP:
    a = df.loc[df.trace_flag == 1, c + "_p"].mean()
    b = df.loc[df.trace_flag == 0, c + "_p"].mean()
    P(f"   {'OK ' if a > b else '역전'} {c+'_p':<18} 침수O {a:.3f} | 침수X {b:.3f} | {a-b:+.3f}")

# ===============================================================
# 4. 검증 — S_phys vs M9(ML) 침수흔적 포착 비교
# ===============================================================
P("\n## 4. 검증: 물리지수 S_phys vs M9(ML OOF)")
y = df.trace_flag.values
try:
    m9 = pd.read_parquet(MOD / "hazard_score.parquet")   # hazard_oof = 공간CV OOF 예측(공정 비교용)
    df = df.merge(m9[["grid_id", "hazard_oof"]].rename(columns={"hazard_oof": "m9_pct"}), on="grid_id", how="left")
    have_m9 = df.m9_pct.notna().mean() > 0.9
except Exception as e:
    have_m9 = False
    P(f"  (M9 점수 로드 실패: {e})")

def topn(score, pct):
    k = max(1, int(len(score) * pct))
    jit = score + np.random.RandomState(RS).rand(len(score)) * max(np.ptp(score), 1e-9) * 1e-9
    top = np.argpartition(-jit, k - 1)[:k]
    return y[top].sum() / max(y.sum(), 1)

P("| 지표 | PR-AUC | Top5% 포착 | Top10% | Top20% | Top25% |")
P("|---|--:|--:|--:|--:|--:|")
for nm, sc in ([("S_phys(물리)", df.s_phys.values)] + ([("M9(ML)", df.m9_pct.values)] if have_m9 else [])):
    ap = average_precision_score(y, sc)
    P(f"| {nm} | {ap:.4f} | {topn(sc,.05):.1%} | {topn(sc,.10):.1%} | {topn(sc,.20):.1%} | {topn(sc,.25):.1%} |")

# 자치구 내부 순위 상관 (신고편향 통제) — 침수흔적 area_ratio 대비
tgt = pd.read_parquet(GG / "03_마스터" / "target_grid.parquet")[["grid_id", "trace_area_ratio"]]
dd = df.merge(tgt, on="grid_id", how="left")
dd["trace_area_ratio"] = dd["trace_area_ratio"].fillna(0.0)
rhos = []
for sgg, g in dd.groupby("sgg_nm"):
    if (g.trace_area_ratio > 0).sum() < 15:
        continue
    rho = spearmanr(g.s_phys, g.trace_area_ratio).correlation
    rhos.append((sgg, rho, int((g.trace_area_ratio > 0).sum())))
rr = pd.DataFrame(rhos, columns=["자치구", "rho", "침수격자수"]).sort_values("rho")
P(f"\n  자치구 내부 Spearman(s_phys, 침수면적비) — 중앙값 {rr.rho.median():+.3f}, "
  f"범위 {rr.rho.min():+.3f}~{rr.rho.max():+.3f}")
P("  하위 3: " + " / ".join(f"{r.자치구} {r.rho:+.2f}" for _, r in rr.head(3).iterrows()))
P("  상위 3: " + " / ".join(f"{r.자치구} {r.rho:+.2f}" for _, r in rr.tail(3).iterrows()))

# ===============================================================
# 5. 저장
# ===============================================================
keep = ["grid_id", "sgg_cd", "sgg_nm", "adm_cd", "adm_nm", "x_cen", "y_cen",
        "hand_m", "s_phys", "s_phys_pct"] + [c + "_p" for c in COMP]
df[keep].to_parquet(MOD / "features_scenario.parquet", index=False)
P(f"\n저장: {MOD/'features_scenario.parquet'} ({df[keep].shape})")
json.dump({"components": {c: {"invert": inv, "weight": w} for c, (inv, w) in COMP.items()},
           "stream_thr_km2": THR_KM2},
          open(MOD / "MS1_설정.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

(REP / "MS1_HAND_물리취약성.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'MS1_HAND_물리취약성.md'}", flush=True)
