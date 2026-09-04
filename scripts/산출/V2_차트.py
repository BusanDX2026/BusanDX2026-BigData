# -*- coding: utf-8 -*-
"""V2. 제출용 성능 차트 — 검증·비교·산출 근거"""
import sys, io, json, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False,
                     "figure.dpi": 150, "axes.grid": True, "grid.alpha": .22,
                     "grid.linewidth": .6, "axes.axisbelow": True})
TEAL, WARM, GREY = "#0d5c58", "#c4703a", "#9aa8a6"

def clean(ax, xg=False):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#ccd4d2"); ax.spines["bottom"].set_color("#ccd4d2")
    ax.grid(axis="x" if xg else "y")
    ax.tick_params(labelsize=9.5, color="#ccd4d2")

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"
OUT = GG / "06_제출자료"; OUT.mkdir(parents=True, exist_ok=True)

d = pd.read_parquet(GG / "05_산출" / "격자_우선순위.parquet").merge(
    pd.read_parquet(GG / "04_모델" / "hazard_score.parquet")[["grid_id", "hazard_oof"]], on="grid_id")
y = d.trace_flag.values
pop = d["pop"].fillna(0).values
tot, par = y.sum(), pop[y == 1].sum()
RS = 42

def topmask(score, p):
    k = int(len(score) * p)
    j = score + np.random.RandomState(RS).rand(len(score)) * max(np.ptp(score), 1e-9) * 1e-9
    m = np.zeros(len(score), bool); m[np.argpartition(-j, k - 1)[:k]] = True
    return m

# ── F05. 커트라인별 재현율·정밀도 ─────────────────────────────
P = [1, 5, 10, 15, 20, 25, 30]
rec = [y[topmask(d.hazard_oof.values, p / 100)].sum() / tot for p in P]
pre = [y[topmask(d.hazard_oof.values, p / 100)].mean() for p in P]
fig, ax = plt.subplots(figsize=(7.6, 4.5))
ax.plot(P, np.array(rec) * 100, "-o", color=TEAL, lw=2.2, ms=7, label="재현율 (침수흔적 포착)")
ax.plot(P, np.array(pre) * 100, "-s", color=WARM, lw=2.2, ms=6.5, label="정밀도")
ax.axhline(3.56, color=GREY, ls="--", lw=1.2)
ax.text(30, 4.8, "무작위 정밀도 3.56%", ha="right", fontsize=8.5, color="#667")
for i, p in enumerate(P):
    if p in (10, 30):
        ax.annotate(f"{rec[i]*100:.1f}%", (p, rec[i] * 100), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=10, fontweight="bold", color=TEAL)
    if p in (1, 10):
        ax.annotate(f"{pre[i]*100:.1f}%", (p, pre[i] * 100), textcoords="offset points",
                    xytext=(0, -18), ha="center", fontsize=10, fontweight="bold", color=WARM)
ax.set_xlabel("선제대응 지정 구역 (상위 %)", fontsize=10)
ax.set_ylabel("비율 (%)", fontsize=10)
ax.set_xticks(P); ax.set_ylim(0, 92)
ax.legend(fontsize=9.5, frameon=False, loc="center right")
ax.set_title("지정 규모별 성능 — 넓힐수록 더 잡지만 정밀도는 떨어진다",
             fontsize=12.5, fontweight="bold", pad=10)
clean(ax); fig.tight_layout()
fig.savefig(OUT / "F05_커트라인별_성능.png", bbox_inches="tight", dpi=300); plt.close()
print("F05 저장")

# ── F06. 피처군 누적 기여 (M9 실측 절제) ──────────────────────
stg = [("물리 6\n표고·경사·TPI\n저지대·하천범람·강우", 0.1921),
       ("＋수문 3\n흐름누적·TWI·수계거리", 0.2610),
       ("＋토지피복 6\n불투수율 등", 0.3012),
       ("＋준설미실시 1\n하수맨홀", 0.3059)]
fig, ax = plt.subplots(figsize=(8.2, 4.6))
ax.bar([s[0] for s in stg], [s[1] for s in stg], color=[GREY, TEAL, TEAL, TEAL], width=.6)
for i, (nm, v) in enumerate(stg):
    ax.text(i, v + .006, f"{v:.4f}", ha="center", fontsize=10.5, fontweight="bold")
    if i:
        ax.text(i, v / 2, f"{v/stg[i-1][1]-1:+.1%}", ha="center", fontsize=10,
                color="white", fontweight="bold")
ax.set_ylabel("PR-AUC (자치구 공간 교차검증)", fontsize=10); ax.set_ylim(0, .345)
ax.set_title("피처군별 누적 기여 — 수문 피처가 최대 기여 (+35.9%)",
             fontsize=12.5, fontweight="bold", pad=10)
ax.tick_params(axis="x", labelsize=8.6)
clean(ax); fig.tight_layout()
fig.savefig(OUT / "F06_개선경로.png", bbox_inches="tight", dpi=300); plt.close()
print("F06 저장")

# ── F07. 기준선 비교 ──────────────────────────────────────────
phys = [c + "_s" for c in json.load(open(GG / "04_모델" / "feature_groups.json", encoding="utf-8"))["HAZARD"]]
_pv = pd.read_parquet(GG / "04_모델" / "features_v4.parquet")[["grid_id"] + phys]
d = d.merge(_pv, on="grid_id", how="left")
base = [("본 모델 (M9)", y[topmask(d.hazard_oof.values, .10)].sum() / tot),
        ("행정 재해위험지구", y[topmask(d.hazdist_flood_active.fillna(0).values
                            + np.random.RandomState(1).rand(len(y)) * 1e-6, .10)].sum() / tot),
        ("무작위", y[topmask(np.random.RandomState(2).rand(len(y)), .10)].sum() / tot),
        ("MCDA 동일가중", y[topmask(d[phys].mean(axis=1).values, .10)].sum() / tot)]
base.sort(key=lambda x: x[1])
fig, ax = plt.subplots(figsize=(7.4, 3.9))
ax.barh([n for n, _ in base], [v * 100 for _, v in base],
        color=[TEAL if n == "본 모델 (M9)" else GREY for n, _ in base], height=.62)
for i, (n, v) in enumerate(base):
    ax.text(v * 100 + 1.2, i, f"{v*100:.1f}%", va="center", fontsize=10.5,
            fontweight="bold" if n == "본 모델 (M9)" else "normal",
            color=TEAL if n == "본 모델 (M9)" else "#555")
ax.set_xlabel("상위 10% 지정 시 침수흔적 포착률 (%)", fontsize=10); ax.set_xlim(0, 68)
ax.set_title("현행 방식 대비 — 행정 지정보다 5배 정확", fontsize=12.5, fontweight="bold", pad=10)
clean(ax, xg=True); fig.tight_layout()
fig.savefig(OUT / "F07_기준선비교.png", bbox_inches="tight", dpi=300); plt.close()
print("F07 저장")

# ── F08. 사건 홀드아웃 ────────────────────────────────────────
ho = [("2011-07-27", 40.9, 48.2), ("2020 대형호우\n3h 114mm", 16.8, 68.0),
      ("2014-08-25\n3h 145mm 극한", 3.6, 33.8)]
fig, ax = plt.subplots(figsize=(7.4, 4.3))
ax.bar([h[0] for h in ho], [h[1] for h in ho], color=[TEAL, TEAL, WARM], width=.55)
for i, h in enumerate(ho):
    ax.text(i, h[1] + 1.1, f"{h[1]:.1f}배", ha="center", fontsize=11.5, fontweight="bold")
    ax.text(i, h[1] / 2, f"상위10%\n{h[2]:.1f}% 포착", ha="center", fontsize=9.5,
            color="white", fontweight="bold")
ax.axhline(1, color="#888", ls="--", lw=1.1)
ax.text(2.42, 2.0, "무작위 = 1배", ha="right", fontsize=8.5, color="#667")
ax.set_ylabel("무작위 대비 리프트 (배)", fontsize=10); ax.set_ylim(0, 46)
ax.set_title("사건 홀드아웃 — 그 호우를 빼고 학습해 그 침수를 맞히는가",
             fontsize=12.5, fontweight="bold", pad=10)
ax.tick_params(axis="x", labelsize=9.5)
clean(ax); fig.tight_layout()
fig.savefig(OUT / "F08_홀드아웃검증.png", bbox_inches="tight", dpi=300); plt.close()
print("F08 저장")

# ── F12. 위험노출인구 포착 (M5 vs M9) ─────────────────────────
ps = [5, 10, 15, 20, 25, 30]
m5 = [pop[topmask(d.priority.values, p / 100) & (y == 1)].sum() / par for p in ps]
m9 = [pop[topmask(d.hazard_oof.values, p / 100) & (y == 1)].sum() / par for p in ps]
fig, ax = plt.subplots(figsize=(7.6, 4.5))
ax.plot(ps, np.array(m5) * 100, "-o", color=TEAL, lw=2.4, ms=7.5, label="M5 MCDA (노출 결합)")
ax.plot(ps, np.array(m9) * 100, "-s", color=GREY, lw=2.2, ms=6.5, label="M9 위험도 단독")
ax.annotate(f"{m5[1]*100:.1f}%", (10, m5[1] * 100), textcoords="offset points", xytext=(7, 9),
            fontsize=11, fontweight="bold", color=TEAL)
ax.annotate(f"{m9[1]*100:.1f}%", (10, m9[1] * 100), textcoords="offset points", xytext=(7, -20),
            fontsize=11, fontweight="bold", color="#667")
ax.axvline(10, color="#ccd4d2", ls=":", lw=1.3)
ax.set_xlabel("선제대응 지정 구역 (상위 %)", fontsize=10)
ax.set_ylabel("침수위험 거주인구 포착률 (%)", fontsize=10)
ax.set_xticks(ps); ax.set_ylim(50, 103)
ax.legend(fontsize=10, frameon=False, loc="lower right")
ax.set_title("사람 기준으로 보면 — 노출을 결합해야 위험인구를 담는다",
             fontsize=12.5, fontweight="bold", pad=10)
clean(ax)
fig.text(.5, -.03, "침수이력 격자 거주인구 279,391명(부산 총인구의 8.5%) 기준",
         ha="center", fontsize=8.5, color="#555")
fig.tight_layout()
fig.savefig(OUT / "F12_위험노출인구_포착.png", bbox_inches="tight", dpi=300); plt.close()
print("F12 저장")

# ── F10. 관측 유발강우 분포 ───────────────────────────────────
act = pd.read_parquet(GG / "02_레이어별" / "침수흔적_활성화강우_격자.parquet")
gA = act[act.n_events >= 2].act_rain_3h.dropna()
gB = act[act.act_only_2014 == 1].act_rain_3h.dropna()
fig, ax = plt.subplots(figsize=(7.6, 4.3))
bins = np.arange(0, 210, 10)
ax.hist(gA, bins=bins, color=TEAL, alpha=.85, label=f"재발형 상습지 (n={len(gA)})")
ax.hist(gB, bins=bins, color=WARM, alpha=.60, label=f"2014급 단발지 (n={len(gB)})")
ax.axvline(gA.median(), color=TEAL, ls="--", lw=1.8)
ax.axvline(gB.median(), color=WARM, ls="--", lw=1.8)
ymax = ax.get_ylim()[1]
ax.text(gA.median() - 4, ymax * .92, f"중앙 {gA.median():.0f}mm", ha="right",
        fontsize=10.5, fontweight="bold", color=TEAL)
ax.text(gB.median() + 4, ymax * .92, f"중앙 {gB.median():.0f}mm", ha="left",
        fontsize=10.5, fontweight="bold", color=WARM)
ax.set_xlabel("관측 유발강우 — 최대 3시간 강우 (mm)", fontsize=10)
ax.set_ylabel("격자 수", fontsize=10)
ax.legend(fontsize=9.5, frameon=False)
ax.set_title("상습 침수지는 더 약한 비에 잠긴다 — 관측으로 확인",
             fontsize=12.5, fontweight="bold", pad=10)
clean(ax); fig.tight_layout()
fig.savefig(OUT / "F10_관측유발강우.png", bbox_inches="tight", dpi=300); plt.close()
print("F10 저장")

# ── F11. 행정동 우선순위 TOP 15 ───────────────────────────────
dg = pd.read_csv(GG / "05_산출" / "행정동_우선순위.csv")
c_pop = "risk_pop" if "risk_pop" in dg.columns else [c for c in dg.columns if "위험" in c][0]
c_nm = "adm_nm" if "adm_nm" in dg.columns else "행정동"
c_sg = "sgg_nm" if "sgg_nm" in dg.columns else "자치구"
t = dg.nlargest(15, c_pop).iloc[::-1]
lab = [f"{a}  {b}" for a, b in zip(t[c_sg], t[c_nm])]
fig, ax = plt.subplots(figsize=(7.8, 6.2))
ax.barh(lab, t[c_pop] / 10000, color=TEAL, height=.68)
for i, v in enumerate(t[c_pop] / 10000):
    ax.text(v + .04, i, f"{v:.1f}만", va="center", fontsize=9.5, color="#333")
ax.set_xlabel("위험 노출인구 (만명) = 상위10% 격자비율 × 인구", fontsize=10)
ax.set_title("선제대응 우선 행정동 TOP 15", fontsize=12.5, fontweight="bold", pad=10)
clean(ax, xg=True); fig.tight_layout()
fig.savefig(OUT / "F11_행정동_우선순위.png", bbox_inches="tight", dpi=300); plt.close()
print("F11 저장")
print("→", OUT)
