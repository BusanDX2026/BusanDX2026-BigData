# -*- coding: utf-8 -*-
"""
M5. MCDA 통합 → 선제대응 우선순위 → 행정동 산출 → 신규/완화 구역 도출

왜 ML 점수를 그대로 안 쓰고 MCDA로 결합하는가
  ML(M4)은 '물이 어디 고이나'(Hazard)만 학습했다. 선제대응구역은 물리적 위험만이 아니라
  '누가 얼마나 노출되나'(Exposure)와 '대응 여력이 있나'(Capacity)를 함께 봐야 정책이 된다.
  또 M4 V3에서 확률 보정 기울기 0.18로 확인됐듯 출력이 절대확률이 아니므로 **백분위 순위**로 결합한다.

⚠ 강서구 보정 (M4 SHAP 발견)
  SHAP에서 lowland3(-0.96)·rain(-0.51)·twi(-0.77) 세 물리량이 '위험 아님' 방향으로 학습됐다.
  전부 강서 삼각주를 가리킨다 — 물리적으로는 취약한데 농경지라 침수흔적 기록이 없어
  모델이 "저지대=안전"으로 잘못 일반화한 것(자치구 내 ρ 0.087로 최하위).
  → 강서구는 ML 점수 대신 **물리 MCDA 점수**를 병기하고, 결과를 별도 표기한다.

- 입력: 04_모델/features_v2.parquet, hazard_score.parquet
- 출력: 05_산출/격자_우선순위.parquet/.gpkg, 행정동_우선순위.csv, 신규완화_구역.csv
"""
import sys, io, json, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"
MOD = GG / "04_모델"; OUT = GG / "05_산출"; OUT.mkdir(parents=True, exist_ok=True)
REP = GG / "_리포트"
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# M5. MCDA 우선순위 · 행정동 산출 · 신규/완화 구역")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

df = pd.read_parquet(MOD / "features_v4.parquet")   # M9 최종 피처셋
hz = pd.read_parquet(MOD / "hazard_score.parquet")[["grid_id", "hazard_raw", "hazard_pct"]]
df = df.merge(hz, on="grid_id", how="left")
GROUPS = json.load(open(MOD / "feature_groups.json", encoding="utf-8"))

# ===============================================================
# 1. 3축 점수 구성 (전부 백분위 = 0~1, 클수록 우선순위↑)
# ===============================================================
P("## 1. 3축 구성")
EXPO = [c + "_s" for c in GROUPS["EXPOSURE"]]
CAP = [c + "_s" for c in GROUPS["CAPACITY"]]
# 노출: 9개 피처가 하나의 상관 클러스터(코드리뷰 r 0.7~0.96) → 단순 평균 후 백분위
df["exposure_pct"] = df[EXPO].mean(axis=1).rank(pct=True)
# 대응취약: 펌프 거리 멀수록/개수 적을수록 취약 (M1에서 방향 정렬 완료)
df["capacity_pct"] = df[CAP].mean(axis=1).rank(pct=True)
# 물리 MCDA(비학습) — 강서 보정 및 비교용
HAZ = ([c + "_s" for c in GROUPS["HAZARD"]] + ["flow_acc_log_s", "twi_s", "dist_stream_m_s"]
       + [c + "_s" for c in ["imperv_ratio","agri_ratio","paddy_ratio","forest_ratio","water_ratio","road_ratio"]]
       + ["mh_no_dredge_ratio_s"])   # M9 최종 16지표
df["hazard_mcda_pct"] = df[HAZ].mean(axis=1).rank(pct=True)
P(f"- Hazard(ML)   : XGBoost 침수감수성 백분위")
P(f"- Hazard(MCDA) : 물리 9지표 동일가중 평균 백분위 (강서 보정·비교용)")
P(f"- Exposure     : {len(EXPO)}개 노출지표 평균 백분위 (인구·건물·지하공간)")
P(f"- Capacity결핍 : {len(CAP)}개 펌프 접근성 평균 백분위")

# ===============================================================
# 2. 가중치 결정 + 민감도 분석
# ===============================================================
P("\n## 2. 가중치 (가산형) 와 민감도")
WEIGHTS = {
    "기본(H.50/E.35/C.15)": (0.50, 0.35, 0.15),
    "위험중심(.70/.20/.10)": (0.70, 0.20, 0.10),
    "노출중심(.35/.50/.15)": (0.35, 0.50, 0.15),
    "균등(.34/.33/.33)":     (0.34, 0.33, 0.33),
}
def score(w, hcol="hazard_pct"):
    wh, we, wc = w
    return wh*df[hcol] + we*df.exposure_pct + wc*df.capacity_pct

P("가중치 조합별 상위 10% 격자의 침수흔적 포착률 / 조합 간 순위상관:")
scs = {k: score(w) for k, w in WEIGHTS.items()}
y = df.trace_flag.values
P("| 가중치 | Top10% 침수포착 | 기본안과 Spearman |")
P("|---|--:|--:|")
b = scs["기본(H.50/E.35/C.15)"]
for k, s in scs.items():
    k10 = int(len(s)*0.10)
    cap = y[np.argsort(-s.values)[:k10]].sum()/y.sum()
    P(f"| {k} | {cap:.1%} | {spearmanr(s, b).statistic:.3f} |")
P("\n→ 가중치를 크게 바꿔도 순위상관 0.9 이상이면 결과가 가중치에 민감하지 않다는 뜻(견고).")
W = WEIGHTS["기본(H.50/E.35/C.15)"]
P(f"- 채택: **기본안 H .50 / E .35 / C .15**. 근거: 물리적 위험이 선행조건이므로 최대,"
  " 노출은 정책 우선순위의 실질 근거라 두 번째, 대응역량은 보정항.")

df["priority_ml"] = score(W, "hazard_pct")
df["priority_mcda"] = score(W, "hazard_mcda_pct")
# 강서구는 ML이 물리량을 역방향 학습 → 물리 MCDA 채택
gangseo = df.sgg_nm == "강서구"
df["priority"] = np.where(gangseo, df.priority_mcda, df.priority_ml)
df["priority_src"] = np.where(gangseo, "MCDA(물리)", "ML")
df["priority_pct"] = df.priority.rank(pct=True)
P(f"- 강서구 {int(gangseo.sum()):,}격자는 물리 MCDA 점수 적용 (사유: M4 SHAP 역방향 학습)")

# ===============================================================
# 3. 검증 — 우선순위가 실제 침수를 잡는가
# ===============================================================
P("\n## 3. 우선순위 검증")
# ── 주 지표: '침수위험 거주인구 포착률' ─────────────────────────────────
#   격자 수 포착률은 정책 지표로 부적절하다. 침수흔적 기록이 저인구 지역에 쏠려 있어
#   (기장군 침수격자당 197명 vs 남구 1,843명 = 9배), 격자 수를 좇으면 사람이 적은 곳을
#   우선하게 된다. 같이 침수될 때 먼저 가야 하는 곳은 사람이 있는 곳이므로
#   **침수이력 격자에 사는 인구를 얼마나 담았는가** 를 주 지표로 삼는다.
pop_v = df["pop"].fillna(0).values
pop_at_risk = pop_v[y == 1].sum()
P(f"- 침수이력 격자 거주인구(= 위험 노출인구): **{pop_at_risk:,.0f}명** "
  f"(부산 총인구의 {pop_at_risk/pop_v.sum():.1%})")
P("")
P("| 상위 | 격자 | **위험노출인구 포착** | 침수격자 포착 | 리프트 | 총 커버인구 |")
P("|---|--:|--:|--:|--:|--:|")
for pct in [0.05, 0.10, 0.20]:
    k = int(len(df) * pct)
    idx = np.argsort(-df.priority.values)[:k]
    m = np.zeros(len(df), bool); m[idx] = True
    P(f"| {pct:.0%} | {k:,} | **{pop_v[m & (y==1)].sum()/pop_at_risk:.1%}** | "
      f"{y[idx].sum()/y.sum():.1%} | {y[idx].sum()/y.sum()/pct:.1f}배 | "
      f"{pop_v[m].sum():,.0f}명 ({pop_v[m].sum()/pop_v.sum():.1%}) |")

# 위험도 단독(M9) 대조 — 노출 결합이 '사람'을 더 담는지
_hz = pd.read_parquet(MOD / "hazard_score.parquet")[["grid_id", "hazard_oof"]]
_d = df[["grid_id"]].merge(_hz, on="grid_id", how="left")
_k = int(len(df) * 0.10)
m9m = np.zeros(len(df), bool); m9m[np.argsort(-_d.hazard_oof.fillna(0).values)[:_k]] = True
m5m = np.zeros(len(df), bool); m5m[np.argsort(-df.priority.values)[:_k]] = True
P("")
P("**상위 10%: 위험도 단독(M9) vs MCDA(노출 결합)**")
P("| 기준 | 위험노출인구 포착 | 침수격자 포착 | 총 커버인구 |")
P("|---|--:|--:|--:|")
for nm, mm in [("M9 위험도 단독", m9m), ("M5 MCDA (0.50/0.35/0.15)", m5m)]:
    P(f"| {nm} | **{pop_v[mm & (y==1)].sum()/pop_at_risk:.1%}** | "
      f"{y[mm].sum()/y.sum():.1%} | {pop_v[mm].sum():,.0f}명 |")
P("")
P("→ 격자 수로는 M9가 앞서지만 **사람 기준으로는 MCDA가 크게 앞선다.** "
  "침수기록이 저인구 지역(기장 31%)에 쏠려 있어 생기는 역전이며, "
  "노출 가중은 편향을 키우는 게 아니라 현실 쪽으로 되돌린다.")


# ===============================================================
# 4. 행정동 우선순위
# ===============================================================
P("\n## 4. 행정동 우선순위 (최종 정책 단위)")
TOP = df.priority_pct >= 0.90
dong = df.groupby(["sgg_nm", "adm_cd", "adm_nm"]).agg(
    격자수=("grid_id", "size"),
    상위10p격자=("priority_pct", lambda s: int((s >= 0.90).sum())),
    평균우선순위=("priority", "mean"),
    인구=("pop", "sum"),
    고령인구=("pop_65", "sum"),
    지하차도=("underpass_n_300m", "max"),
    침수흔적격자=("trace_flag", "sum"),
    재해지정=("hazdist_flood_active", "max"),
).reset_index()
dong["상위10p비율"] = dong.상위10p격자 / dong.격자수
# 행정동 순위 = 상위격자 비율 × 인구 가중 (사람이 많은 곳 우선)
dong["위험인구"] = dong.상위10p비율 * dong.인구
dong = dong.sort_values("위험인구", ascending=False).reset_index(drop=True)
dong["순위"] = np.arange(1, len(dong)+1)
dong.to_csv(OUT / "행정동_우선순위.csv", index=False, encoding="utf-8-sig")
P("**상위 20개 행정동** (위험인구 = 상위10% 격자비율 × 인구)")
P("| 순위 | 자치구 | 행정동 | 상위10%격자 | 비율 | 인구 | 위험인구 | 침수흔적 | 재해지정 |")
P("|--:|---|---|--:|--:|--:|--:|--:|:-:|")
for _, r in dong.head(20).iterrows():
    P(f"| {r.순위} | {r.sgg_nm} | {r.adm_nm} | {r.상위10p격자} | {r.상위10p비율:.0%} | "
      f"{r.인구:,.0f} | {r.위험인구:,.0f} | {int(r.침수흔적격자)} | {'O' if r.재해지정 else ''} |")

# ===============================================================
# 5. 신규 편입 / 완화 검토 — 행정 지정 대비 차이 (프로젝트 차별점)
# ===============================================================
P("\n## 5. 행정 지정 대비 — 신규 권고 / 완화 검토")
P("우리 지수 상위 10% 격자와 현행 자연재해위험개선지구(유효+침수) 를 교차한다.")
ours = df.priority_pct >= 0.90
admin = df.hazdist_flood_active == 1
P(f"- 우리 상위10% {int(ours.sum()):,}격자 / 행정지정 {int(admin.sum()):,}격자 / 교집합 {int((ours&admin).sum()):,}격자")
P(f"- 행정지정 중 우리도 상위10%인 비율: **{(ours&admin).sum()/max(admin.sum(),1):.1%}**")

cand = df[ours & ~admin].groupby(["sgg_nm", "adm_nm"]).agg(
    격자=("grid_id", "size"), 인구=("pop", "sum"), 침수흔적=("trace_flag", "sum"),
    평균우선순위=("priority", "mean")).reset_index()
P(f"\n### 신규 편입 권고 후보")
P(f"- 느슨한 기준(5격자 이상)이면 {int((cand.격자>=5).sum())}개 행정동 — 부산 전체 206개의 "
  f"{int((cand.격자>=5).sum())/206:.0%}라 **정책적으로 무의미**. 실행 가능한 기준으로 조인다.")
NEW_CRIT = dict(격자=20, 인구=3000, 침수흔적=5)
P(f"- 채택 기준: 상위10% 격자 **≥{NEW_CRIT['격자']}개**(2ha 이상 집적) ∧ 인구 **≥{NEW_CRIT['인구']:,}명**(실질 노출) "
  f"∧ 침수흔적 **≥{NEW_CRIT['침수흔적']}격자**(실증 근거) ∧ 미지정")
new = cand[(cand.격자 >= NEW_CRIT["격자"]) & (cand.인구 >= NEW_CRIT["인구"]) &
           (cand.침수흔적 >= NEW_CRIT["침수흔적"])].sort_values(
    ["침수흔적", "인구"], ascending=False).reset_index(drop=True)
P(f"\n**최종 신규 편입 권고: {len(new)}개 행정동**")
P("| 자치구 | 행정동 | 상위10%격자 | 인구 | 침수흔적격자 | 평균우선순위 |")
P("|---|---|--:|--:|--:|--:|")
for _, r in new.iterrows():
    P(f"| {r.sgg_nm} | {r.adm_nm} | {r.격자} | {r.인구:,.0f} | {int(r.침수흔적)} | {r.평균우선순위:.3f} |")
P(f"\n⚠ 기장군 주의: 침수흔적 격자 886개(전체의 31%)로 인구 대비 과다. 전처리 코드리뷰 FIND-3의 "
  "**지자체별 조사편향** 가능성이 높아 기장군 후보는 현장 확인 우선순위를 낮춘다.")

rel = df[admin & (df.priority_pct < 0.70)].groupby(["sgg_nm", "adm_nm"]).agg(
    격자=("grid_id", "size"), 인구=("pop", "sum"), 침수흔적=("trace_flag", "sum"),
    평균우선순위=("priority", "mean")).reset_index()
rel = rel[rel.침수흔적 == 0].sort_values("격자", ascending=False).reset_index(drop=True)
P(f"\n### 완화 검토 후보 (행정지정 ∧ 우리 하위 70% 미만 ∧ 침수흔적 0) — {len(rel)}개 행정동")
P("| 자치구 | 행정동 | 격자 | 인구 | 침수흔적격자 | 평균우선순위 |")
P("|---|---|--:|--:|--:|--:|")
for _, r in rel.head(10).iterrows():
    P(f"| {r.sgg_nm} | {r.adm_nm} | {r.격자} | {r.인구:,.0f} | {int(r.침수흔적)} | {r.평균우선순위:.3f} |")
P("\n※ 인구가 0~수명인 행정동이 섞이는 것은 우리 지수가 노출(인구)을 35% 반영하기 때문이다. "
  "농경지·산업지의 자산 피해는 이 지수가 과소평가하므로, 해당 구역의 완화 판단은 보류한다.")
P("\n⚠ '완화'는 해제 권고가 아니라 **재검토 대상**이다. 재해위험지구는 정비사업 완료로 위험이 실제 낮아진 경우도 포함하므로"
  " 현장 확인이 전제다. 또 지오코딩 정밀도 한계(7/71건 구중심근사, 전처리 L-6)를 감안해야 한다.")
pd.concat([new.assign(구분="신규편입권고"), rel.assign(구분="완화검토")]).to_csv(
    OUT / "신규완화_구역.csv", index=False, encoding="utf-8-sig")

# ===============================================================
# 6. 산출 저장
# ===============================================================
grid = gpd.read_file(GG / "01_격자" / "grid_100m.gpkg")[["grid_id", "geometry"]]
keep = ["grid_id", "sgg_cd", "sgg_nm", "adm_cd", "adm_nm", "hazard_pct", "hazard_mcda_pct",
        "exposure_pct", "capacity_pct", "priority", "priority_pct", "priority_src",
        "pop", "trace_flag", "hazdist_flood_active"]
df[keep].to_parquet(OUT / "격자_우선순위.parquet", index=False)
gpd.GeoDataFrame(df[keep].merge(grid, on="grid_id"), crs=5186).to_file(
    OUT / "격자_우선순위.gpkg", driver="GPKG")
P(f"\n- 산출: 05_산출/격자_우선순위.parquet/.gpkg, 행정동_우선순위.csv, 신규완화_구역.csv")

(REP / "M5_MCDA_산출.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'M5_MCDA_산출.md'}", flush=True)
