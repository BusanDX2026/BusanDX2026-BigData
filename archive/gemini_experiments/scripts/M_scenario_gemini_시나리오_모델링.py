# -*- coding: utf-8 -*-
"""
M_scenario_gemini. 4대 강우 시나리오 기반 동적 침수 위험도 모델링
- 시나리오 4종:
    S1: 호우주의보 (일 100mm, ~30mm/hr) — 국지 저지대 침수 시작
    S2: 호우경보 (일 160mm, ~50mm/hr) — 평년 연최대 강우 (현재 기본값)
    S3: 극한호우 (일 250mm, ~80mm/hr) — 2014.8.25 부산 폭우 수준 (대규모 침수)
    S4: 100년 빈도 초극한 (일 350mm, ~100+mm/hr) — 기후변화 최악 시나리오
- 분석 내용:
    1) 시나리오별 침수 위험도(감수성) 추론 및 고위험 격자 팽창(Expansion)
    2) 시나리오별 실제 과거 침수흔적(전체 및 2014 호우) 포착률 곡선 산출
    3) 시나리오별 위험 노출 인구 및 건물 면적 변화
    4) 행정동별 침수 시작 임계 강우량(Critical Tipping Point) 도출
- 산출물:
    05_산출/시나리오별_위험도_gemini.parquet,
    05_산출/행정동별_침수임계강우_gemini.csv,
    _리포트/M_scenario_gemini_결과.md
"""
import sys, io, json, time, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"
MOD = GG / "04_모델"
OUT = GG / "05_산출"; OUT.mkdir(parents=True, exist_ok=True)
REP = GG / "_리포트"
RS = 42
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# M_scenario_gemini. 강우 시나리오 기반 동적 침수 위험도 모델링")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

df = pd.read_parquet(MOD / "features_v3_gemini.parquet")
cfg = json.load(open(MOD / "M3_gemini_최종설정.json", encoding="utf-8"))
FEATS = cfg["features"]
PARAMS = cfg["params"]
SPW = cfg["scale_pos_weight"]

P(f"기본 격자: {len(df):,}행 · 피처 {len(FEATS)}개")

# 1. 모델 전체 학습
X_orig = df[FEATS].values
y = df.trace_flag.values
final_mdl = XGBClassifier(
    **PARAMS, scale_pos_weight=SPW, eval_metric="aucpr",
    tree_method="hist", reg_lambda=1.0, n_jobs=-1, random_state=RS, verbosity=0
).fit(X_orig, y)

# 2. 4대 시나리오 강우 피처 생성
# 원본 rain_annmax_mm 의 p5, p95 (약 151.44, 167.39)
p5 = np.percentile(df["rain_annmax_mm"], 5)
p95 = np.percentile(df["rain_annmax_mm"], 95)
base_mean_rain = df["rain_annmax_mm"].mean()  # 약 159 mm

SCENARIOS = {
    "S1": {"name": "호우주의보 (일 100mm / 시우량 ~30mm)", "target_mm": 100.0},
    "S2": {"name": "호우경보 (일 160mm / 시우량 ~50mm)",   "target_mm": 160.0},
    "S3": {"name": "대형 극한호우 (일 250mm / 시우량 ~80mm, 2014년 사상)", "target_mm": 250.0},
    "S4": {"name": "100년 빈도 초극한호우 (일 350mm / 시우량 ~100mm)", "target_mm": 350.0},
}

rain_idx = FEATS.index("rain_annmax_mm_s")

P("## 1. 4대 강우 시나리오 정의 및 추론")
scen_results = {}
total_flood = y.sum()
flood_2014 = ((df.trace_flag == 1) & (df.trace_last_year == 2014)).sum()
total_pop = df["pop"].sum()

for s_id, s_info in SCENARIOS.items():
    ratio = s_info["target_mm"] / base_mean_rain
    sim_rain = df["rain_annmax_mm"] * ratio
    # 정규화 피처 변환 (시나리오 확장에 따른 상대 강우값)
    sim_rain_s = np.clip((sim_rain - p5) / (p95 - p5), 0.0, 2.5)  # 상한선 확장 허용
    
    # 시나리오 입력 복사
    X_sim = X_orig.copy()
    X_sim[:, rain_idx] = sim_rain_s
    
    # 추론
    prob = final_mdl.predict_proba(X_sim)[:, 1]
    pct = pd.Series(prob).rank(pct=True).values
    
    df[f"prob_{s_id}"] = prob
    df[f"pct_{s_id}"] = pct
    
    # 고위험 격자 기준: 확률 > 0.5 (또는 상위 15% 위험선)
    high_risk = prob >= 0.5
    n_high = high_risk.sum()
    high_flood = y[high_risk].sum()
    high_f2014 = ((high_risk) & (df.trace_flag == 1) & (df.trace_last_year == 2014)).sum()
    high_pop = df.loc[high_risk, "pop"].sum()
    
    # Top 10% 및 Top 25% 포착률
    k10 = int(len(df) * 0.10)
    order = np.argsort(-prob)
    c10 = y[order[:k10]].sum() / total_flood
    k25 = int(len(df) * 0.25)
    c25 = y[order[:k25]].sum() / total_flood
    
    scen_results[s_id] = {
        "name": s_info["name"],
        "target_mm": s_info["target_mm"],
        "high_risk_grids": n_high,
        "high_risk_pct": n_high / len(df),
        "flood_captured": high_flood,
        "flood_recall": high_flood / total_flood,
        "flood_2014_recall": high_f2014 / flood_2014,
        "exposed_pop": high_pop,
        "pop_rate": high_pop / total_pop,
        "top10_recall": c10,
        "top25_recall": c25,
    }
    P(f"- [{s_id}] {s_info['name']}: 고위험 격자 {n_high:,}개 ({(n_high/len(df)):.1%}) | 침수흔적 포착률 {high_flood/total_flood:.1%} (2014 호우 {high_f2014/flood_2014:.1%})")

# ---------------------------------------------------------------
# 2. 시나리오별 결과 비교 테이블
# ---------------------------------------------------------------
P("\n## 2. 시나리오별 침수 위험 확산(Expansion) 비교")
P("| 시나리오 | 상응 강우량 | 고위험 격자수 (확률≥50%) | **전체 침수 포착률** | **2014 대호우 포착률** | 위험 노출 인구 | Top 25% 포착률 |")
P("|---|---|--:|--:|--:|--:|--:|")
for s_id, r in scen_results.items():
    P(f"| **{s_id}** | {r['target_mm']:.0f} mm/day | {r['high_risk_grids']:,} ({r['high_risk_pct']:.1%}) | **{r['flood_recall']:.1%}** | **{r['flood_2014_recall']:.1%}** | {int(r['exposed_pop']):,}명 ({r['pop_rate']:.1%}) | **{r['top25_recall']:.1%}** |")

# ---------------------------------------------------------------
# 3. 행정동별 침수 임계 강우량(Tipping Point) 분석
# ---------------------------------------------------------------
P("\n## 3. 행정동별 침수 임계 강우량(Tipping Point) 분석")
dong_scen = df.groupby(["sgg_nm", "adm_cd", "adm_nm"]).agg(
    total_grids=("grid_id", "size"),
    pop_total=("pop", "sum"),
    trace_grids=("trace_flag", "sum"),
    high_S1=("prob_S1", lambda s: int((s >= 0.5).sum())),
    high_S2=("prob_S2", lambda s: int((s >= 0.5).sum())),
    high_S3=("prob_S3", lambda s: int((s >= 0.5).sum())),
    high_S4=("prob_S4", lambda s: int((s >= 0.5).sum())),
).reset_index()

dong_scen["ratio_S1"] = dong_scen["high_S1"] / dong_scen["total_grids"]
dong_scen["ratio_S2"] = dong_scen["high_S2"] / dong_scen["total_grids"]
dong_scen["ratio_S3"] = dong_scen["high_S3"] / dong_scen["total_grids"]
dong_scen["ratio_S4"] = dong_scen["high_S4"] / dong_scen["total_grids"]

# 임계 등급 판정:
# 1) 초취약: S1(100mm)에서 이미 고위험 격자 비율 >= 20%
# 2) 경계: S2(160mm)에서 고위험 격자 비율 >= 20%
# 3) 극한취약: S3(250mm) 이상이어야 고위험 격자 비율 >= 20%
# 4) 안전방어: S4(350mm)에서도 고위험 격자 비율 < 20%
def classify_tipping(r):
    if r["ratio_S1"] >= 0.20:
        return "1단계 초취약 (100mm 이하 즉각 침수)"
    elif r["ratio_S2"] >= 0.20:
        return "2단계 경계 (160mm 집중호우 침수)"
    elif r["ratio_S3"] >= 0.20:
        return "3단계 극한취약 (250mm 대호우 침수)"
    elif r["ratio_S4"] >= 0.20:
        return "4단계 초극한취약 (350mm 미증유 침수)"
    else:
        return "5단계 고지대 방어동"

dong_scen["tipping_class"] = dong_scen.apply(classify_tipping, axis=1)

P(f"- 행정동별 임계 침수 단계 분포: {dong_scen['tipping_class'].value_counts().to_dict()}")

P("\n### [초취약 행정동 TOP 10] (시간당 30mm/일 100mm 호우 시 즉각 고위험 전이)")
P("| 순위 | 자치구 | 행정동 | 총격자 | S1위험격자 | S1위험비율 | 인구 | 침수흔적격자 |")
P("|--:|---|---|--:|--:|--:|--:|--:|")
top_s1 = dong_scen.sort_values("ratio_S1", ascending=False).head(10)
for i, (_, r) in enumerate(top_s1.iterrows(), 1):
    P(f"| {i} | {r['sgg_nm']} | {r['adm_nm']} | {r['total_grids']} | {r['high_S1']} | **{r['ratio_S1']:.1%}** | {int(r['pop_total']):,}명 | {int(r['trace_grids'])} |")

P("\n### [극한호우 급변 행정동 TOP 5] (S1~S2는 견디다 S3(250mm)에서 침수 격자 폭증)")
dong_scen["jump_S3"] = dong_scen["high_S3"] - dong_scen["high_S1"]
top_jump = dong_scen.sort_values("jump_S3", ascending=False).head(5)
P("| 순위 | 자치구 | 행정동 | S1위험격자 | S3위험격자 | 위험격자 증가수 | 인구 |")
P("|--:|---|---|--:|--:|--:|--:|")
for i, (_, r) in enumerate(top_jump.iterrows(), 1):
    P(f"| {i} | {r['sgg_nm']} | {r['adm_nm']} | {r['high_S1']} | {r['high_S3']} | **+{r['jump_S3']}격자** | {int(r['pop_total']):,}명 |")

# ---------------------------------------------------------------
# 4. 산출물 저장
# ---------------------------------------------------------------
out_parquet = OUT / "시나리오별_위험도_gemini.parquet"
cols_to_save = ["grid_id", "sgg_nm", "adm_nm", "x_cen", "y_cen", "pop", "trace_flag",
                "prob_S1", "prob_S2", "prob_S3", "prob_S4",
                "pct_S1", "pct_S2", "pct_S3", "pct_S4"]
df[cols_to_save].to_parquet(out_parquet, index=False)
P(f"\n저장: {out_parquet} ({df[cols_to_save].shape})")

dong_scen.to_csv(OUT / "행정동별_침수임계강우_gemini.csv", index=False, encoding="utf-8-sig")
P(f"저장: {OUT/'행정동별_침수임계강우_gemini.csv'} ({len(dong_scen)}개 행정동)")

(REP / "M_scenario_gemini_결과.md").write_text("\n".join(log), encoding="utf-8")
print(f"==> 리포트: {REP/'M_scenario_gemini_결과.md'}")
