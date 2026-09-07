# -*- coding: utf-8 -*-
"""
MS2. 격자별 '활성화 강우 등급' 산출 → 강우 시나리오별 선제대응 우선순위

■ 핵심 아이디어 (제미나이 방식의 실패를 우회)
  제미나이는 ML 모델의 rain 피처를 전역으로 키워 시나리오를 만들었다. 그 피처는 공간지문이라
  강우↑ → 위험격자↓ 라는 비단조가 나왔다(무효).
  대신 여기서는 강우강도 축을 '침수 재발성(recurrence)'에서 뽑는다.
    - 재발 격자(2회 이상 침수) : median HAND 0.0m, imperv 0.76 → '상시(약한비에도) 침수'
    - 2014 단발 격자           : median HAND 0.18m, imperv 0.40 → '극한호우에서만 침수'
    (HAND 차이 Mann-Whitney p=3e-30, MS1 진단)
  즉 물리·도시형태 피처로 '이 격자가 재발형인가'를 학습하면 = '낮은 강우에 활성화되는가'가 된다.

■ 3단계 활성화 강우 등급 (KMA 특보 기준 앵커)
  T1 상시취약   ~ 100 mm/day (호우주의보, 3h 60 / 12h 110mm)  — 재발형 도시침수 상습지
  T2 집중호우취약 ~ 180 mm/day (호우경보,  3h 90 / 12h 180mm)  — 2020-class
  T3 극한호우취약 ~ 280 mm/day (2014.8.25 ≈ 일 260mm, ~100년빈도) — 미증유 확산
  T0 방어       : 어느 등급에서도 비활성

■ 시나리오 우선순위 = 누적 포함(T1 ⊂ T1+T2 ⊂ T1+T2+T3)
  → 강우가 커지면 구역이 '추가'만 되므로 포착률 단조 증가가 구조적으로 보장됨(제미나이 버그 불가능)
  각 등급 내부 공간순위는 M9 hazard(공간 판별력 PR-AUC 0.24)로 매김

■ 입력 : 04_모델/features_v4.parquet, features_scenario.parquet, hazard_score.parquet
         03_마스터/target_grid.parquet, baseline_grid.parquet, 02_레이어별/홍수위험_grid.parquet
■ 출력 : 05_산출/시나리오_활성화등급.parquet, 시나리오_행정동_우선순위.csv
         _리포트/MS2_활성화강우_시나리오.md
"""
import sys, io, json, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"
MOD = GG / "04_모델"
OUT = GG / "05_산출"; OUT.mkdir(parents=True, exist_ok=True)
REP = GG / "_리포트"
RS = 42
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# MS2. 활성화 강우 등급 → 강우 시나리오별 선제대응 우선순위")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

# ---------------------------------------------------------------
# 0. 데이터 결합
# ---------------------------------------------------------------
df = pd.read_parquet(MOD / "features_v4.parquet")   # 이미 hazdist_flood_active, fluv_area_ratio, trace_* 포함
sc = pd.read_parquet(MOD / "features_scenario.parquet")
hz = pd.read_parquet(MOD / "hazard_score.parquet")[["grid_id", "hazard_oof"]]
tg = pd.read_parquet(GG / "03_마스터" / "target_grid.parquet")[["grid_id", "trace_area_ratio"]]
fr = pd.read_parquet(GG / "02_레이어별" / "홍수위험_grid.parquet")[["grid_id", "pluv_area_ratio"]]
# 재발 정의를 MS3/S11 과 통일 (코드리뷰 #10): trace_count 는 같은 폭우의 중복 폴리곤까지 세므로
#   서로 다른 날짜의 사건 수 n_events 를 쓴다.
ne = pd.read_parquet(GG / "02_레이어별" / "침수흔적_활성화강우_격자.parquet")[["grid_id", "n_events", "act_only_2014"]]
df = (df.merge(sc.drop(columns=[c for c in sc.columns if c in df.columns and c != "grid_id"]), on="grid_id")
        .merge(hz, on="grid_id").merge(tg, on="grid_id", how="left").merge(fr, on="grid_id", how="left")
        .merge(ne, on="grid_id", how="left"))
df["n_events"] = df["n_events"].fillna(0)
df["act_only_2014"] = df["act_only_2014"].fillna(0).astype(int)
df["trace_area_ratio"] = df["trace_area_ratio"].fillna(0.0)
df["hazdist_flood_active"] = df["hazdist_flood_active"].fillna(0).astype(int)
for c in ["fluv_area_ratio", "pluv_area_ratio"]:
    df[c] = df[c].fillna(0.0)
P(f"격자 {len(df):,} · 침수흔적 {int(df.trace_flag.sum()):,} · 200년하천범람 {(df.fluv_area_ratio>0).sum():,} · 재해위험(침수) {df.hazdist_flood_active.sum():,}")

# ---------------------------------------------------------------
# 1. 활성화 강우 점수 A_score — '낮은 강우에 활성화되는가'의 물리 지표
#    투명한 물리 결합만 사용 (ML 래퍼 없음). HAND가 핵심:
#    격자가 최근접 배수로보다 얼마나 낮은가 = 침수 시작에 필요한 강우/수위.
#    검증은 '이 점수가 재발형(=낮은 강우 침수) vs 2014단발(=극한 침수)을 가르는가'로 수행.
# ---------------------------------------------------------------
P("\n## 1. 활성화 강우 점수 A_score (투명 물리 결합)")
# A_score = 0.50 HAND역(낮을수록↑) + 0.30 TWI + 0.20 s_phys_pct
df["A_score"] = 0.50 * df.hand_m_p + 0.30 * df.twi_p + 0.20 * df.s_phys_pct
df["A_score_pct"] = df.A_score.rank(pct=True)

# --- 검증 1: A_score 가 재발/2014단발을 가르는가 (침수흔적 격자, 라벨 미사용 → 순환 없음) ---
fl = df[df.trace_flag == 1].copy()
fl["y_recur"] = (fl.n_events >= 2).astype(int)
auc_pool = roc_auc_score(fl.y_recur, fl.A_score)
aucs_in = []
for sgg, g in fl.groupby("sgg_nm"):
    if g.y_recur.nunique() == 2 and len(g) >= 20:
        aucs_in.append(roc_auc_score(g.y_recur, g.A_score))
auc_in = float(np.mean(aucs_in))
P(f"- 침수흔적 격자 {len(fl):,} (재발2회+ {fl.y_recur.sum():,} / 단발 {(fl.y_recur==0).sum():,})")
P(f"- A_score의 재발성 분리 AUC: 전체 {auc_pool:.3f} / 자치구내부 평균 {auc_in:.3f} (n_sgg={len(aucs_in)})")
P(f"  → 0.5~0.6대 = '약한비 침수'와 '극한 침수'를 **방향은 맞으나 느슨하게** 구분. 등급은 정밀 임계가 아닌 광역 밴드로 해석.")
P(f"- HAND≤1m 비율: 재발격자 {(fl.loc[fl.y_recur==1,'hand_m']<=1).mean():.0%} vs 2014단발격자 {(fl.loc[fl.act_only_2014==1,'hand_m']<=1).mean():.0%}")

# ---------------------------------------------------------------
# 2. 3단계 등급
# ---------------------------------------------------------------
P("\n## 2. 활성화 강우 3단계 등급")

# 시나리오 대응 격자 '우주(universe)' = 어떤 강우에서든 침수 가능성 있는 격자
#   = M9 상위 30% ∪ 침수흔적 ∪ 200년하천범람(면적비>0.1) ∪ 도시침수100년 ∪ 재해위험(침수)
df["m9_pct"] = df.hazard_oof.rank(pct=True)
universe = ((df.m9_pct >= 0.70) | (df.trace_flag == 1) | (df.fluv_area_ratio > 0.10) |
            (df.pluv_area_ratio > 0) | (df.hazdist_flood_active == 1))
df["in_universe"] = universe.astype(int)
P(f"- 대응 우주(universe): {int(universe.sum()):,}격자 ({universe.mean():.1%}) "
  f"— 침수흔적 포착 {df.loc[universe,'trace_flag'].sum()/df.trace_flag.sum():.1%}")

# 등급 점수 = 0.70·M9(공간 판별력) + 0.30·A_score(활성화 강우, HAND 주도)
#   순수 A_score 3분위는 M9 대비 Top10% 포착이 28% vs 53%로 급락 → 공간정밀도 유지 위해 M9 주도.
#   A_score는 상시침수 물리표지(HAND 낮음)를 상단으로 끌어올리고 고HAND를 T3로 미는 보정 역할.
df["tier_score"] = 0.70 * df.m9_pct + 0.30 * df.A_score_pct
u = df[universe].copy()
q1, q2 = u.tier_score.quantile([1/3, 2/3])
def tier(r):
    if not r.in_universe:
        return 0
    if r.tier_score >= q2:
        return 1
    if r.tier_score >= q1:
        return 2
    return 3
df["tier"] = df.apply(tier, axis=1)

# 200년 하천범람 우세 격자는 universe 규칙(fluv>0.10)에 이미 전부 포함되어 tier>=1 이 보장된다.
#   (구버전의 `tier==0` 조건부 T3 편입 규칙은 논리상 절대 발동하지 않는 죽은 코드였다 — 코드리뷰 #6)
assert (df.loc[df.fluv_area_ratio > 0.30, "tier"] > 0).all(), "200년 하천범람 격자가 대응구역 밖에 있음"
P(f"- 200년 하천범람 우세 {int((df.fluv_area_ratio>0.30).sum()):,}격자 — universe 규칙으로 전부 포함 확인")

SCEN = {1: ("T1 상시취약", "≈100 mm/day (호우주의보)"),
        2: ("T2 집중호우취약", "≈180 mm/day (호우경보)"),
        3: ("T3 극한호우취약", "≈280 mm/day (2014급·100년빈도)")}
for t, (nm, rr) in SCEN.items():
    n = int((df.tier == t).sum())
    P(f"  {nm:<14} {rr:<28} {n:>6,}격자 (누적 {int((df.tier.between(1,t)).sum()):,})")
P(f"  T0 방어        {'':<28} {int((df.tier==0).sum()):>6,}격자")

# ---------------------------------------------------------------
# 3. 검증 A — 재발/2014단발 격자가 등급에 올바르게 분포하나
# ---------------------------------------------------------------
P("\n## 3. 검증 A: 침수흔적 격자의 등급 분포 (재발형은 T1, 2014단발은 T2·T3로 가야 정상)")
fl2 = df[df.trace_flag == 1].copy()
fl2["grp"] = np.where(fl2.n_events >= 2, "재발(2회+)",
              np.where(fl2.act_only_2014 == 1, "2014단발", "기타단발"))
ct = pd.crosstab(fl2.grp, fl2.tier, normalize="index").round(3)
P("| 그룹 | T1 | T2 | T3 | T0(미포착) |")
P("|---|--:|--:|--:|--:|")
for g in ["재발(2회+)", "기타단발", "2014단발"]:
    if g in ct.index:
        r = ct.loc[g]
        P(f"| {g} | {r.get(1,0):.1%} | {r.get(2,0):.1%} | {r.get(3,0):.1%} | {r.get(0,0):.1%} |")

# ---------------------------------------------------------------
# 4. 검증 B — 시나리오별 누적 침수흔적 포착률 (단조 증가 확인)
# ---------------------------------------------------------------
P("\n## 4. 검증 B: 강우 시나리오별 누적 포착률 (단조 증가 = 제미나이 버그 없음)")
y = df.trace_flag.values
is2014 = ((df.trace_flag == 1) & (df.act_only_2014 == 1)).values
recur = ((df.trace_flag == 1) & (df.n_events >= 2)).values
tot, tot14, totR = y.sum(), is2014.sum(), recur.sum()
totpop = df["pop"].sum()
P("| 시나리오 | 강우 | 누적 대응격자 | 면적 | 전체 포착 | 재발형 포착 | 2014급 포착 | 노출인구 |")
P("|---|---|--:|--:|--:|--:|--:|--:|")
prev = 0
for t, (nm, rr) in SCEN.items():
    m = df.tier.between(1, t).values
    n = int(m.sum())
    cap = y[m].sum() / tot
    cap14 = is2014[m].sum() / tot14
    capR = recur[m].sum() / totR
    pop = df.loc[m, "pop"].sum()
    assert cap >= prev - 1e-9, "포착률 비단조!"
    prev = cap
    P(f"| {nm} | {rr.split('(')[0].strip()} | {n:,} | {n*0.01:.0f} km² | **{cap:.1%}** | {capR:.1%} | {cap14:.1%} | {int(pop):,} ({pop/totpop:.0%}) |")

# ---------------------------------------------------------------
# 5. 검증 C — universe 공간 판별력 (M9 대비 손실 없나) & 자치구 내부 상관
# ---------------------------------------------------------------
P("\n## 5. 검증 C: 공간 판별력 · 자치구 내부 순위상관")
def topn(score, pct):
    """전체 격자 중 상위 pct 를 골랐을 때의 침수흔적 포착률."""
    k = max(1, int(len(score) * pct))
    jit = score + np.random.RandomState(RS).rand(len(score)) * 1e-9
    return y[np.argpartition(-jit, k - 1)[:k]].sum() / tot
# 종합 시나리오 점수 = tier_score (등급 정의와 동일). 등급 순위 부여 후 등급 내부도 tier_score.
df["scen_score"] = np.where(df.in_universe == 1, df.tier_score, -1.0)
P("  (등급 정의 점수 = 0.70·M9 + 0.30·A_score)")
for pct in [0.05, 0.10, 0.20, 0.25]:
    a = topn(df.scen_score.values, pct)
    b = topn(df.hazard_oof.values, pct)
    P(f"  Top{pct:.0%}  시나리오점수 {a:.1%}  |  M9 단독 {b:.1%}")
rhos = []
for sgg, g in df.groupby("sgg_nm"):
    if (g.trace_area_ratio > 0).sum() < 15:
        continue
    rhos.append((sgg, spearmanr(g.tier_score, g.trace_area_ratio).correlation))
rr2 = pd.DataFrame(rhos, columns=["자치구", "rho"]).sort_values("rho")
P(f"  자치구내 Spearman(tier_score, 침수면적비) 중앙값 {rr2.rho.median():+.3f} "
  f"(하위 {rr2.iloc[0].자치구} {rr2.iloc[0].rho:+.2f} / 상위 {rr2.iloc[-1].자치구} {rr2.iloc[-1].rho:+.2f})")

# ---------------------------------------------------------------
# 6. 행정동별 시나리오 우선순위
# ---------------------------------------------------------------
P("\n## 6. 행정동별 시나리오 선제대응 우선순위")
df["risk_pop_cell"] = df["pop"] * df["m9_pct"]
rows = []
for (sgg, adm, nm), g in df.groupby(["sgg_nm", "adm_cd", "adm_nm"]):
    d = dict(자치구=sgg, 행정동=nm, 총격자=len(g), 인구=int(g["pop"].sum()),
             T1=int((g.tier == 1).sum()), T2=int((g.tier == 2).sum()), T3=int((g.tier == 3).sum()),
             침수흔적=int(g.trace_flag.sum()),
             노출인구_T1=int(g.loc[g.tier == 1, "pop"].sum()),
             노출인구_누적T2=int(g.loc[g.tier.between(1, 2), "pop"].sum()),
             평균위험=round(g.m9_pct.mean(), 3))
    d["1단계강도"] = d["T1"] / d["총격자"]
    rows.append(d)
dong = pd.DataFrame(rows)
dong["scen_priority"] = (dong.노출인구_T1 * 1.0 + dong.노출인구_누적T2 * 0.4).round(0)
dong = dong.sort_values("scen_priority", ascending=False).reset_index(drop=True)

P("\n### 호우주의보(T1, ~100mm) 단계 즉시대응 상위 15개 행정동")
P("| 순위 | 자치구 | 행정동 | T1격자 | T1강도 | T1노출인구 | 침수흔적 | 평균위험 |")
P("|--:|---|---|--:|--:|--:|--:|--:|")
for i, r in dong.head(15).iterrows():
    P(f"| {i+1} | {r.자치구} | {r.행정동} | {r.T1} | {r['1단계강도']:.0%} | {r.노출인구_T1:,} | {r.침수흔적} | {r.평균위험:.2f} |")

P("\n### 호우주의보→경보 상향 시 신규 편입 상위 10개 행정동 (T1 적으나 T2 급증)")
# 경보 상향 시 신규 편입 격자수 = 그 동의 T2 격자수 (T1 은 이미 편입돼 있음)
for i, r in dong[dong.T1 <= 3].sort_values("T2", ascending=False).head(10).reset_index(drop=True).iterrows():
    P(f"- {r.자치구} {r.행정동}: T1 {r.T1}격자 → 경보 시 +{r.T2}격자 신규 (인구 {r.인구:,})")

# ---------------------------------------------------------------
# 7. 저장
# ---------------------------------------------------------------
save_cols = ["grid_id", "sgg_nm", "adm_nm", "x_cen", "y_cen", "pop", "trace_flag", "trace_count",
             "hand_m", "s_phys", "A_score", "A_score_pct", "m9_pct", "tier_score", "in_universe", "tier",
             "fluv_area_ratio", "hazdist_flood_active"]
df[save_cols].to_parquet(OUT / "시나리오_활성화등급.parquet", index=False)
P(f"\n저장: {OUT/'시나리오_활성화등급.parquet'} ({df[save_cols].shape})")
dong.to_csv(OUT / "시나리오_행정동_우선순위.csv", index=False, encoding="utf-8-sig")
P(f"저장: {OUT/'시나리오_행정동_우선순위.csv'} ({len(dong)}개 행정동)")
json.dump({"scenarios": {str(k): {"name": v[0], "rain": v[1]} for k, v in SCEN.items()},
           "A_score_formula": "0.50*hand_m_p + 0.30*twi_p + 0.20*s_phys_pct",
           "tier_score_formula": "0.70*m9_pct + 0.30*A_score_pct",
           "A_recur_separation_auc": {"pooled": round(auc_pool, 3), "within_sgg": round(auc_in, 3)},
           "universe_rule": "m9_pct>=0.70 | trace | fluv>0.10 | pluv>0 | hazdist_flood",
           "tier_cut_quantiles": [round(float(q1), 4), round(float(q2), 4)]},
          open(MOD / "MS2_설정.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

(REP / "MS2_활성화강우_시나리오.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'MS2_활성화강우_시나리오.md'}", flush=True)
