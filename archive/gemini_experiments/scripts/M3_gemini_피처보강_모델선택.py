# -*- coding: utf-8 -*-
"""
M3_gemini. 피처 보강(도심 불투수율·집수압 스트레스) → 모델 재학습 및 성능 비교
- 기존 물리 피처(9개): elev_min, slope_mean, tpi, lowland3_ratio, fluv_area_ratio,
                       rain_annmax_mm, flow_acc_log, twi, dist_stream_m
- 신규 물리 피처(2개 채택):
    1) imperv_bldg_ratio_s : 격자 내 건물 바닥면적(건축면적) 비율 (0~1) — 도심 불투수 표면 유출 지표
    2) urban_runoff_stress_s: twi_s * imperv_bldg_ratio_s — 지형습윤(웅덩이)과 불투수의 결합 유출 스트레스
- 검증: 자치구 GroupKFold(5-fold) 공간 교차검증 (M3 기존 0.2017 대비 비교)
- 산출: 04_모델/features_v3_gemini.parquet, 04_모델/M3_gemini_최종설정.json, _리포트/M3_gemini_피처보강.md
"""
import sys, io, json, time, glob, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"
RAW = ROOT / "공공데이터" / "raw"
MOD = GG / "04_모델"
REP = GG / "_리포트"
RS = 42
CELL_AREA = 100.0 * 100.0  # 10,000 m2
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# M3_gemini. 도심 불투수율 & 복합 집수압 피처 보강 모델링")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

# 기존 v2 피처 로드
df = pd.read_parquet(MOD / "features_v2.parquet")
P(f"기존 features_v2: {df.shape}")

# ---------------------------------------------------------------
# 1. 건물 기반 불투수율/건폐율 (imperv_bldg_ratio) 추출
# ---------------------------------------------------------------
P("\n## 1. 건물 기반 불투수율(건폐율) 지표 산출")
t0 = time.perf_counter()
bshp = glob.glob(str(RAW / "03_건물_GIS건물통합정보_부산" / "**" / "AL_D010_26_20260809.shp"), recursive=True)[0]
b = gpd.read_file(bshp, columns=["A12", "A14", "A26", "A27"], encoding="cp949").to_crs(5186)
b["A12"] = pd.to_numeric(b["A12"], errors="coerce").fillna(0.0)
b["A14"] = pd.to_numeric(b["A14"], errors="coerce").fillna(0.0)
b["_foot"] = b.geometry.area
# 건축면적: A12와 footprint geometry 면적 중 유효한 값 선택 (외피 오류 방어)
b["footprint"] = np.where((b["A12"] > 0) & (b["A12"] < b["_foot"] * 3), b["A12"], b["_foot"])
b["footprint"] = np.minimum(b["footprint"], 40000.0)  # 격자 초과 방어
b["rep"] = b.geometry.representative_point()

grid = gpd.read_file(GG / "01_격자" / "grid_100m.gpkg")[["grid_id", "geometry"]].to_crs(5186)
bpts = gpd.GeoDataFrame(b[["footprint"]], geometry=b["rep"].values, crs=5186)
j = gpd.sjoin(bpts, grid[["grid_id", "geometry"]], how="inner", predicate="within")
b_agg = j.groupby("grid_id")["footprint"].sum().reset_index()
b_agg = b_agg.rename(columns={"footprint": "bldg_footprint_area"})

df = df.merge(b_agg, on="grid_id", how="left")
df["bldg_footprint_area"] = df["bldg_footprint_area"].fillna(0.0)
# 건폐율 / 불투수율: 격자 10,000 m2 대비 건물 바닥면적 비율 (상한 1.0)
df["imperv_bldg_ratio"] = np.clip(df["bldg_footprint_area"] / CELL_AREA, 0.0, 1.0)
P(f"- 건물 불투수율 산출 완료 ({time.perf_counter()-t0:.1f}s)")
P(f"  건물 바닥면적 평균 {df.bldg_footprint_area.mean():.1f}m² | 불투수율 중앙값 {df.imperv_bldg_ratio.median():.3f} | 최대 {df.imperv_bldg_ratio.max():.3f}")

# ---------------------------------------------------------------
# 2. 신규 피처 정규화 (log1p + robust min-max 5~95%)
# ---------------------------------------------------------------
P("\n## 2. 신규 피처 정규화")
def norm_robust(s, use_log=False):
    v = s.values.astype(float)
    if use_log:
        v = np.log1p(np.maximum(v, 0))
    p5, p95 = np.percentile(v, [5, 95])
    if p95 <= p5:
        p5, p95 = v.min(), max(v.max(), v.min() + 1e-9)
    return np.clip((v - p5) / (p95 - p5), 0.0, 1.0)

df["imperv_bldg_ratio_s"] = norm_robust(df["imperv_bldg_ratio"], use_log=True)

# 상호작용 피처: 도심 집수압 = TWI(웅덩이/수문습윤) * 불투수율
df["urban_runoff_stress"] = df["twi_s"] * df["imperv_bldg_ratio_s"]
df["urban_runoff_stress_s"] = norm_robust(df["urban_runoff_stress"], use_log=False)

# 방향 검증: 침수흔적 격자에서 점수가 높아야 정상
for c in ["imperv_bldg_ratio_s", "urban_runoff_stress_s"]:
    a = df.loc[df.trace_flag == 1, c].mean()
    z = df.loc[df.trace_flag == 0, c].mean()
    P(f"  {'OK ' if a > z else '역전'} {c:<24} 침수O {a:.3f} | 침수X {z:.3f} | {a-z:+.3f}")

# ---------------------------------------------------------------
# 3. XGBoost 모델 재학습 및 자치구 GroupKFold5 비교
# ---------------------------------------------------------------
P("\n## 3. 자치구 GroupKFold(5-fold) 공간 교차검증 비교")
BASE_FEATS = [
    "elev_min_s", "slope_mean_s", "tpi_s", "lowland3_ratio_s",
    "fluv_area_ratio_s", "rain_annmax_mm_s", "flow_acc_log_s",
    "twi_s", "dist_stream_m_s"
]
NEW_FEATS = [
    "imperv_bldg_ratio_s", "urban_runoff_stress_s"
]
ALL_PHYS_FEATS = BASE_FEATS + NEW_FEATS
P(f"- 기존 물리 피처: {len(BASE_FEATS)}개")
P(f"- 확장 물리 피처 (gemini): {len(ALL_PHYS_FEATS)}개 (+{len(NEW_FEATS)}개)")

y = df.trace_flag.values
groups = df.sgg_cd.values
gkf = list(GroupKFold(n_splits=5).split(df, y, groups))
spw = (len(y) - y.sum()) / y.sum()

def topn_rate(y_true, s, pct):
    k = max(1, int(len(s) * pct))
    o = s + np.random.RandomState(RS).rand(len(s)) * max(np.ptp(s), 1e-9) * 1e-9
    return y_true[np.argpartition(-o, k - 1)[:k]].sum() / max(y_true.sum(), 1)

def eval_cv(feats, name):
    X = df[feats].values
    oof = np.zeros(len(y))
    t_start = time.perf_counter()
    for tr, te in gkf:
        mdl = XGBClassifier(
            max_depth=4, n_estimators=800, learning_rate=0.02,
            min_child_weight=20, subsample=0.7, colsample_bytree=0.7,
            scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist",
            reg_lambda=1.0, n_jobs=-1, random_state=RS, verbosity=0
        )
        mdl.fit(X[tr], y[tr])
        oof[te] = mdl.predict_proba(X[te])[:, 1]
    elapsed = time.perf_counter() - t_start
    pr = average_precision_score(y, oof)
    roc = roc_auc_score(y, oof)
    t5 = topn_rate(y, oof, 0.05)
    t10 = topn_rate(y, oof, 0.10)
    t20 = topn_rate(y, oof, 0.20)
    t25 = topn_rate(y, oof, 0.25)
    P(f"[{name}]")
    P(f"  PR-AUC: {pr:.4f} | ROC-AUC: {roc:.4f}")
    P(f"  포착률: Top5% {t5:.1%} | Top10% {t10:.1%} | Top20% {t20:.1%} | Top25% {t25:.1%}")
    P(f"  학습시간: {elapsed:.1f}s")
    return oof, pr, roc, t5, t10, t20, t25

oof_base, pr_b, roc_b, t5_b, t10_b, t20_b, t25_b = eval_cv(BASE_FEATS, "기존 M3 (9개 물리 피처)")
oof_gem, pr_g, roc_g, t5_g, t10_g, t20_g, t25_g = eval_cv(ALL_PHYS_FEATS, "gemini 확장 (11개 물리 피처)")

P("\n## 4. 피처 보강 효과 요약")
P(f"- PR-AUC: {pr_b:.4f} → **{pr_g:.4f}** ({((pr_g/pr_b)-1)*100:+.2f}%)")
P(f"- ROC-AUC: {roc_b:.4f} → **{roc_g:.4f}** ({((roc_g/roc_b)-1)*100:+.2f}%)")
P(f"- Top 10% 포착률: {t10_b:.1%} → **{t10_g:.1%}** ({(t10_g-t10_b)*100:+.1f}%p)")
P(f"- Top 20% 포착률: {t20_b:.1%} → **{t20_g:.1%}** ({(t20_g-t20_b)*100:+.1f}%p)")
P(f"- Top 25% 포착률: {t25_b:.1%} → **{t25_g:.1%}** ({(t25_g-t25_b)*100:+.1f}%p)")

# 최종 전체 학습
final_gem = XGBClassifier(
    max_depth=4, n_estimators=800, learning_rate=0.02,
    min_child_weight=20, subsample=0.7, colsample_bytree=0.7,
    scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist",
    reg_lambda=1.0, n_jobs=-1, random_state=RS, verbosity=0
).fit(df[ALL_PHYS_FEATS].values, y)

df["hazard_gemini_raw"] = final_gem.predict_proba(df[ALL_PHYS_FEATS].values)[:, 1]
df["hazard_gemini_oof"] = oof_gem

# 산출물 저장
out_parquet = MOD / "features_v3_gemini.parquet"
df.to_parquet(out_parquet, index=False)
P(f"\n저장: {out_parquet} ({df.shape})")

cfg_gemini = {
    "features": ALL_PHYS_FEATS,
    "params": {
        "max_depth": 4, "n_estimators": 800, "learning_rate": 0.02,
        "min_child_weight": 20, "subsample": 0.7, "colsample_bytree": 0.7
    },
    "scale_pos_weight": float(spw)
}
json.dump(cfg_gemini, open(MOD / "M3_gemini_최종설정.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

(REP / "M3_gemini_피처보강.md").write_text("\n".join(log), encoding="utf-8")
print(f"==> 리포트: {REP/'M3_gemini_피처보강.md'}")
