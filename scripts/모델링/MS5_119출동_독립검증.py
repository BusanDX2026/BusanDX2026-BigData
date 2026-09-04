# -*- coding: utf-8 -*-
"""
MS5. 119 배수출동 기반 독립 검증

■ 왜 필요한가
  지금까지 모든 성능은 **침수흔적도 하나의 타깃**에 대해 측정됐다. 그런데 침수흔적은
  자치구별 조사 강도가 달라(기장 31%) 편향이 크고, 사건이 18개뿐이며 자치구에 쏠려 있다(#29).
  119 소방출동은 **완전히 다른 출처·다른 수집 경로**이므로, 우리 우선순위가 실제
  침수 대응 수요를 예측하는지 **교차검증**할 수 있다.

■ 데이터 (S12 수집)
  강우일(전지점 최대3h >= 30mm) 100일 × 2020~2025, 총 72,182건 중
  침수 신호 = `지원출동(배수)` + `지원출동(풍수해)` = **1,034건 / 59일 / 16개 자치구**
  ※ `자연재해`(6,383건)는 제외한다 — 태풍 강풍 피해가 지배적이다.
     2020-09-03 태풍 마이삭: 자연재해 1,947건인데 배수는 2건.
     강우 상관도 배수(Pearson +0.50) > 자연재해(+0.29).

■ 공간 매칭
  119 `dsraddr` 은 법정동, 우리 격자는 행정동. 이름 정규화로 결합한다.
     서대신동1가 → 서대신동 ← 서대신1동 ,  거제1동 → 거제동
  매칭률 **95.4%** (986/1,034건). 미매칭 48건은 강서 농촌부(송정·봉림 등 1행정동이
  여러 법정동을 포괄)로, 제외하고 그 사실을 명시한다.

■ 핵심 통제
  배수출동은 사람이 신고해야 생기므로 **인구에 자동으로 비례**한다. 따라서 단순 상관은
  의미가 없다. 인구를 통제한 **부분상관**이 진짜 검증이다.

■ 입출력
  IN : 02_레이어별/119출동_강우일_원시.parquet, 강우_AWS시간_long.parquet
       05_산출/격자_우선순위.parquet, 04_모델/hazard_score.parquet
  OUT: 05_산출/119검증_행정동.csv, _리포트/MS5_119출동_독립검증.md
"""
import sys, io, re, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"
LAY = GG / "02_레이어별"; MOD = GG / "04_모델"; OUT = GG / "05_산출"; REP = GG / "_리포트"
RAIN_TH = 30.0
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# MS5. 119 배수출동 기반 독립 검증")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

def base_name(n):
    """법정동·행정동 이름을 공통 base 로 정규화. 서대신동1가/서대신1동 → 서대신동"""
    n = str(n).strip()
    n = re.sub(r"\d+가$", "", n)
    n = re.sub(r"\d+동$", "동", n)
    return n

# ---------------------------------------------------------------
# 1. 119 배수출동 집계
# ---------------------------------------------------------------
P("## 1. 119 배수출동 집계")
df = pd.read_parquet(LAY / "119출동_강우일_원시.parquet")
rain = pd.read_parquet(LAY / "강우_AWS시간_long.parquet"); rain["tm"] = pd.to_datetime(rain.tm)
r = rain[(rain.tm >= "2020-01-01") & (rain.tm < "2026-09-05")]
piv = r.pivot_table(index="tm", columns="stn", values="rain_mm", aggfunc="max").sort_index().asfreq("h").fillna(0)
day3 = piv.rolling(3, min_periods=1).sum().max(axis=1).pipe(lambda s: s.groupby(s.index.normalize()).max())
days = set(d.strftime("%Y%m%d") for d in day3[day3 >= RAIN_TH].index)

df["ymd"] = df.regtime.astype(str).str[:8]
n_raw = len(df)
df = df[df.ymd.isin(days)]                      # regtime 부분문자열 매칭 오염 제거
P(f"- 원시 {n_raw:,}건 → 대상 강우일 소속 {len(df):,}건 (부분문자열 오염 {n_raw-len(df)}건 제거)")

FLOOD = ["지원출동(배수)", "지원출동(풍수해)"]
f = df[df.dsrclscd.isin(FLOOD)].copy()
f["sgg"] = f.dsraddr.astype(str).str.split().str[1]
f["dong"] = f.dsraddr.astype(str).str.split().str[2]
f["base"] = f.dong.map(base_name)
P(f"- 침수 신호({'+'.join(FLOOD)}): **{len(f):,}건** / {f.ymd.nunique()}일 / {f.sgg.nunique()}개 자치구")
P(f"- 참고: 제외한 `자연재해` {int((df.dsrclscd=='자연재해').sum()):,}건 — 태풍 강풍피해 지배")

d119 = f.groupby(["sgg", "base"]).size().reset_index(name="dispatch")

# ---------------------------------------------------------------
# 2. 우리 산출물을 행정동 base 로 집계
# ---------------------------------------------------------------
P("\n## 2. 격자 산출물 → 행정동 base 집계")
pr = pd.read_parquet(OUT / "격자_우선순위.parquet")
hz = pd.read_parquet(MOD / "hazard_score.parquet")[["grid_id", "hazard_oof"]]
g = pr.merge(hz, on="grid_id", how="left")
g["base"] = g.adm_nm.map(base_name)
g["m9_pct"] = g.hazard_oof.rank(pct=True)
g["pri_pct"] = g.priority.rank(pct=True)
k10 = int(len(g) * 0.10)
g["is_top10_pri"] = (g.priority.rank(ascending=False) <= k10).astype(int)
g["is_top10_m9"] = (g.hazard_oof.rank(ascending=False) <= k10).astype(int)

agg = g.groupby(["sgg_nm", "base"]).agg(
    격자=("grid_id", "size"), 인구=("pop", "sum"), 침수흔적=("trace_flag", "sum"),
    M5우선순위=("pri_pct", "mean"), M9위험도=("m9_pct", "mean"),
    M5상위10비율=("is_top10_pri", "mean"), M9상위10비율=("is_top10_m9", "mean"),
).reset_index().rename(columns={"sgg_nm": "sgg"})

m = agg.merge(d119, on=["sgg", "base"], how="left")
m["dispatch"] = m.dispatch.fillna(0)
matched = d119.merge(agg[["sgg", "base"]], on=["sgg", "base"], how="inner").dispatch.sum()
P(f"- 행정동 base 단위 {len(m)}개 · 매칭된 출동 {int(matched):,}/{len(f):,}건 ({matched/len(f):.1%})")
P(f"- 미매칭 {int(len(f)-matched)}건은 강서 농촌부 등(1행정동이 여러 법정동 포괄) → 제외")
P(f"- 배수출동이 1건 이상인 base: {int((m.dispatch>0).sum())}개 / {len(m)}개")

# ---------------------------------------------------------------
# 3. 검증 — 우리 우선순위가 실제 출동을 예측하는가
# ---------------------------------------------------------------
P("\n## 3. 검증 A: 상관 (전체 base)")
def sp(a, b):
    v = spearmanr(a, b)
    return v.correlation, v.pvalue
P("| 예측지표 | Spearman | p-value |")
P("|---|--:|--:|")
CAND = [("M5 MCDA 우선순위", "M5우선순위"), ("M9 위험도 단독", "M9위험도"),
        ("M5 상위10% 격자비율", "M5상위10비율"), ("침수흔적 격자수", "침수흔적"),
        ("**인구 (대조군)**", "인구"), ("격자수=면적 (대조군)", "격자")]
for nm, c in CAND:
    rho, p = sp(m[c], m.dispatch)
    P(f"| {nm} | {rho:+.3f} | {p:.1e} |")

P("\n## 4. 검증 B: **인구 통제 부분상관** (핵심)")
P("배수출동은 사람이 신고해야 생기므로 인구에 자동 비례한다. 인구를 통제해도 남는 상관이 진짜 신호다.")
def partial_sp(x, y, z):
    """z 를 통제한 x,y 의 부분 스피어만 (순위 잔차 상관)"""
    rx = pd.Series(x).rank(); ry = pd.Series(y).rank(); rz = pd.Series(z).rank()
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return spearmanr(ex, ey)
P("")
P("| 예측지표 | 단순 Spearman | **인구 통제 후** | p-value |")
P("|---|--:|--:|--:|")
for nm, c in CAND[:4]:
    r0, _ = sp(m[c], m.dispatch)
    v = partial_sp(m[c], m.dispatch, m.인구)
    P(f"| {nm} | {r0:+.3f} | **{v.correlation:+.3f}** | {v.pvalue:.1e} |")

P("\n## 5. 검증 C: 상위 N% 행정동이 실제 출동을 얼마나 담나")
P("| 상위 | 동수 | M5 우선순위 기준 | M9 위험도 기준 | 인구 기준(대조) |")
P("|---|--:|--:|--:|--:|")
tot_d = m.dispatch.sum()
for pct in [0.10, 0.20, 0.30, 0.50]:
    k = max(1, int(len(m) * pct))
    row = [f"| {pct:.0%} | {k} "]
    for c in ["M5우선순위", "M9위험도", "인구"]:
        top = m.nlargest(k, c)
        row.append(f"| {top.dispatch.sum()/tot_d:.1%} ")
    P("".join(row) + "|")

P("\n## 6. 검증 D: 인구당 출동 (신고 성향 통제한 순수 침수 취약도)")
mm = m[m.인구 >= 3000].copy()
mm["출동_만명당"] = mm.dispatch / mm.인구 * 10000
P(f"- 인구 3,000명 이상 base {len(mm)}개 대상")
for nm, c in [("M5 MCDA 우선순위", "M5우선순위"), ("M9 위험도 단독", "M9위험도"), ("침수흔적 격자수", "침수흔적")]:
    rho, p = sp(mm[c], mm.출동_만명당)
    P(f"  {nm:<20} vs 인구만명당 출동: Spearman **{rho:+.3f}** (p={p:.1e})")

P("\n## 7. 배수출동 상위 행정동 vs 우리 순위")
m["출동순위"] = m.dispatch.rank(ascending=False, method="min").astype(int)
m["M5순위"] = m.M5우선순위.rank(ascending=False, method="min").astype(int)
top = m.nlargest(15, "dispatch")
P("| 자치구 | 행정동 | 배수출동 | 인구 | M5 순위 | 침수흔적 |")
P("|---|---|--:|--:|--:|--:|")
for _, x in top.iterrows():
    P(f"| {x.sgg} | {x.base} | {int(x.dispatch)} | {int(x.인구):,} | {int(x.M5순위)}/{len(m)} | {int(x.침수흔적)} |")

m.sort_values("dispatch", ascending=False).to_csv(OUT / "119검증_행정동.csv", index=False, encoding="utf-8-sig")
P(f"\n저장: {OUT/'119검증_행정동.csv'} ({m.shape})")
(REP / "MS5_119출동_독립검증.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'MS5_119출동_독립검증.md'}", flush=True)
