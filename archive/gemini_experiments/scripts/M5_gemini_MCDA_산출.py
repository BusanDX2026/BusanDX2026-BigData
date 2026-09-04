# -*- coding: utf-8 -*-
"""
M5_gemini. MCDA 통합 → 선제대응 우선순위 → 다단계 방재(80% 포착선) → 행정동 산출
- 입력: 04_모델/features_v3_gemini.parquet, 04_모델/hazard_score_gemini.parquet
- 핵심:
    1) 3축 결합: Hazard_gemini(50%) + Exposure(35%) + Capacity결핍(15%)
    2) 다단계 방재 경보선(Top 5%, 10%, 15%, 20%, 25%, 30%)에 따른 침수흔적 포착률 검증 (80% 도달선 규명)
    3) 강서구 하이브리드 보정 (물리 MCDA 채택)
    4) 206개 행정동 선제대응 우선순위 갱신 및 신규 편입 권고 행정동 도출
- 출력: 05_산출/격자_우선순위_gemini.parquet/.gpkg, 행정동_우선순위_gemini.csv, _리포트/M5_gemini_산출.md
"""
import sys, io, json, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"
MOD = GG / "04_모델"
OUT = GG / "05_산출"; OUT.mkdir(parents=True, exist_ok=True)
REP = GG / "_리포트"
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# M5_gemini. MCDA 다기준 통합 & 다단계 방재 80% 포착선 산출")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

df = pd.read_parquet(MOD / "features_v3_gemini.parquet")
hz = pd.read_parquet(MOD / "hazard_score_gemini.parquet")[["grid_id", "hazard_gemini_raw", "hazard_gemini_pct"]]
df = df.merge(hz, on="grid_id", how="left")
GROUPS = json.load(open(MOD / "feature_groups.json", encoding="utf-8"))

# ---------------------------------------------------------------
# 1. 3축 백분위 순위 점수 구성
# ---------------------------------------------------------------
P("## 1. 3대 축 백분위 점수 구성")
EXPO = [c + "_s" for c in GROUPS["EXPOSURE"]]
CAP = [c + "_s" for c in GROUPS["CAPACITY"]]
HAZ = [c + "_s" for c in GROUPS["HAZARD"]] + ["flow_acc_log_s", "twi_s", "dist_stream_m_s",
                                              "imperv_bldg_ratio_s", "urban_runoff_stress_s"]

df["exposure_pct"] = df[EXPO].mean(axis=1).rank(pct=True)
df["capacity_pct"] = df[CAP].mean(axis=1).rank(pct=True)
df["hazard_mcda_pct"] = df[HAZ].mean(axis=1).rank(pct=True)

P(f"- Hazard(ML_gemini): XGBoost 11피처 침수감수성 백분위")
P(f"- Hazard(MCDA)    : 물리 11지표 동일가중 백분위 (강서구 보정용)")
P(f"- Exposure        : {len(EXPO)}개 노출지표 평균 백분위")
P(f"- Capacity결핍    : {len(CAP)}개 펌프 접근성 평균 백분위")

# ---------------------------------------------------------------
# 2. 가중 결합 및 강서구 하이브리드 보정
# ---------------------------------------------------------------
P("\n## 2. 가중 결합 (위험 50% + 노출 35% + 대응결핍 15%)")
WH, WE, WC = 0.50, 0.35, 0.15
df["priority_ml"] = WH * df["hazard_gemini_pct"] + WE * df["exposure_pct"] + WC * df["capacity_pct"]
df["priority_mcda"] = WH * df["hazard_mcda_pct"] + WE * df["exposure_pct"] + WC * df["capacity_pct"]

gangseo = df.sgg_nm == "강서구"
df["priority"] = np.where(gangseo, df.priority_mcda, df.priority_ml)
df["priority_src"] = np.where(gangseo, "MCDA(물리)", "ML_gemini")
df["priority_pct"] = df.priority.rank(pct=True)
P(f"- 강서구 {int(gangseo.sum()):,}격자 물리 MCDA 보정 적용 완료")

# ---------------------------------------------------------------
# 3. 다단계 방재 경보선별 포착률 (80% 달성선 규명)
# ---------------------------------------------------------------
P("\n## 3. 다단계 방재 기준별 침수흔적 포착률 & 인구 커버리지 (80% 달성 분석)")
P("| 방재 관리 단계 | 격자수 | 격자 비율 | 침수흔적 포착수 | **침수흔적 포착률** | 보호 인구수 | 인구 커버율 |")
P("|---|--:|--:|--:|--:|--:|--:|")

y = df.trace_flag.values
total_flood = y.sum()
total_pop = df["pop"].sum()

CUTOFFS = [
    ("1단계 [심각 대응] (Top 5%)", 0.05),
    ("2단계 [경계 대응] (Top 10%)", 0.10),
    ("3단계 [주의 경보] (Top 15%)", 0.15),
    ("4단계 [예비 주의] (Top 20%)", 0.20),
    ("5단계 [광역 관리] (Top 25%)", 0.25),
    ("6단계 [종합 방어] (Top 30%)", 0.30),
]

order = np.argsort(-df.priority.values)
for label, pct in CUTOFFS:
    k = int(len(df) * pct)
    top_idx = order[:k]
    f_cnt = y[top_idx].sum()
    f_rate = f_cnt / total_flood
    p_cnt = df.iloc[top_idx]["pop"].sum()
    p_rate = p_cnt / total_pop
    mark = " 🎯 (80% 돌파!)" if f_rate >= 0.80 else ""
    P(f"| {label} | {k:,} | {pct:.0%} | {int(f_cnt):,} | **{f_rate:.1%}**{mark} | {int(p_cnt):,}명 | {p_rate:.1%} |")

# ---------------------------------------------------------------
# 4. 행정동 선제대응 우선순위 산출
# ---------------------------------------------------------------
P("\n## 4. 행정동 선제대응 우선순위 TOP 20")
k10 = int(len(df) * 0.10)
top10_set = set(order[:k10])
df["is_top10"] = [i in top10_set for i in range(len(df))]

dong_agg = df.groupby(["sgg_nm", "adm_cd", "adm_nm"]).agg(
    total_grids=("grid_id", "size"),
    top10_grids=("is_top10", "sum"),
    pop_total=("pop", "sum"),
    flood_grids=("trace_flag", "sum"),
    mean_priority=("priority", "mean"),
).reset_index()

dong_agg["top10_ratio"] = dong_agg["top10_grids"] / dong_agg["total_grids"]
dong_agg["risk_pop"] = dong_agg["pop_total"] * dong_agg["top10_ratio"]
dong_agg = dong_agg.sort_values("risk_pop", ascending=False).reset_index(drop=True)

P("| 순위 | 자치구 | 행정동 | 상위10%격자 | 격자비율 | 인구 | **위험노출인구** | 침수흔적격자 | 평균우선순위 |")
P("|--:|---|---|--:|--:|--:|--:|--:|--:|")
for i, r in dong_agg.head(20).iterrows():
    P(f"| {i+1} | {r['sgg_nm']} | {r['adm_nm']} | {r['top10_grids']} | {r['top10_ratio']:.1%} | {int(r['pop_total']):,} | **{int(r['risk_pop']):,}명** | {int(r['flood_grids'])} | {r['mean_priority']:.3f} |")

# ---------------------------------------------------------------
# 5. 산출물 저장
# ---------------------------------------------------------------
out_parquet = OUT / "격자_우선순위_gemini.parquet"
df.to_parquet(out_parquet, index=False)
P(f"\n저장: {out_parquet} ({df.shape})")

dong_agg.to_csv(OUT / "행정동_우선순위_gemini.csv", index=False, encoding="utf-8-sig")
P(f"저장: {OUT/'행정동_우선순위_gemini.csv'} ({len(dong_agg)}개 행정동)")

(REP / "M5_gemini_산출.md").write_text("\n".join(log), encoding="utf-8")
print(f"==> 리포트: {REP/'M5_gemini_산출.md'}")
