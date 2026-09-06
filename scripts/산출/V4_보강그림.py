# -*- coding: utf-8 -*-
"""V4. 보강 그림 3종 — SHAP 해석 / 자치구별 편향통제 / 강우 시나리오 누적포착"""
import sys, io, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

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

# ── F13. SHAP 기여도 ─────────────────────────────────────────
sh = pd.read_csv(GG / "04_모델" / "M9_shap.csv").head(10).iloc[::-1]
KO = {"elev_min_s": "최저표고", "dist_stream_m_s": "수계 거리", "mh_no_dredge_ratio_s": "하수맨홀 준설미실시",
      "forest_ratio_s": "산림 비율", "tpi_s": "지형위치지수(TPI)", "twi_s": "지형습윤지수(TWI)",
      "rain_annmax_mm_s": "연최대 강우", "imperv_ratio_s": "불투수율", "flow_acc_log_s": "흐름누적",
      "slope_mean_s": "평균 경사", "fluv_area_ratio_s": "하천범람 면적비", "lowland3_ratio_s": "저지대(≤3m) 비율",
      "water_ratio_s": "수계 비율", "agri_ratio_s": "농경지 비율", "paddy_ratio_s": "논 비율",
      "road_ratio_s": "도로 비율"}
lab = [KO.get(f, f) for f in sh.feature]
col = [TEAL if d > 0 else WARM for d in sh.dir]
fig, ax = plt.subplots(figsize=(7.8, 5.2))
ax.barh(lab, sh.share * 100, color=col, height=.68)
for i, (v, dd) in enumerate(zip(sh.share * 100, sh.dir)):
    ax.text(v + .4, i, f"{v:.1f}%", va="center", fontsize=9.5, color="#333")
ax.set_xlabel("SHAP 기여율 (%)", fontsize=10)
ax.set_title("무엇이 침수 감수성을 만드나 — 상위 10개 피처",
             fontsize=12.5, fontweight="bold", pad=10)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=TEAL, label="값↑ → 위험↑"),
                   Patch(facecolor=WARM, label="값↑ → 위험↓")],
          fontsize=9, frameon=False, loc="lower right")
clean(ax, xg=True); fig.tight_layout()
fig.text(.5, -.02, "최저표고·수계거리·TPI 등 상대지형이 상위 — 절대 저지대보다 '주변 대비 낮음'이 결정적",
         ha="center", fontsize=8.5, color="#555")
fig.savefig(OUT / "F13_SHAP기여도.png", bbox_inches="tight", dpi=300); plt.close()
print("F13 저장")

# ── F14. 자치구별 편향통제 검증 ──────────────────────────────
d = pd.read_parquet(GG / "05_산출" / "격자_우선순위.parquet").merge(
    pd.read_parquet(GG / "04_모델" / "hazard_score.parquet")[["grid_id", "hazard_oof"]], on="grid_id")
tg = pd.read_parquet(GG / "03_마스터" / "target_grid.parquet")[["grid_id", "trace_area_ratio"]]
d = d.merge(tg, on="grid_id", how="left")
d["trace_area_ratio"] = d.trace_area_ratio.fillna(0)
rows = []
for s, g in d.groupby("sgg_nm"):
    if (g.trace_area_ratio > 0).sum() >= 15:
        rows.append((s, spearmanr(g.hazard_oof, g.trace_area_ratio).correlation,
                     int((g.trace_flag == 1).sum())))
r = pd.DataFrame(rows, columns=["구", "rho", "n"]).sort_values("rho")
fig, ax = plt.subplots(figsize=(7.8, 5.4))
ax.barh(r.구, r.rho, color=[WARM if v < 0.15 else TEAL for v in r.rho], height=.68)
for i, (v, n) in enumerate(zip(r.rho, r.n)):
    ax.text(v + .012, i, f"{v:+.2f}", va="center", fontsize=9.5, color="#333")
med = r.rho.median()
ax.axvline(med, color="#667", ls="--", lw=1.3)
ax.text(med + .008, len(r) - .4, f"중앙값 {med:+.2f}", fontsize=9, color="#556")
ax.set_xlabel("자치구 내부 Spearman (예측 위험도 ↔ 실제 침수면적비)", fontsize=10)
ax.set_title("조사편향 통제 — 자치구 '안에서'도 순위가 맞는가",
             fontsize=12.5, fontweight="bold", pad=10)
ax.set_xlim(0, max(r.rho) * 1.18)
clean(ax, xg=True); fig.tight_layout()
fig.text(.5, -.02, "16개 자치구 전부 ρ>0 — 자치구별 조사 강도가 달라도 구 내부 우선순위는 유효",
         ha="center", fontsize=8.5, color="#555")
fig.savefig(OUT / "F14_자치구내_편향통제.png", bbox_inches="tight", dpi=300); plt.close()
print("F14 저장")

# ── F15. 강우 시나리오 누적 포착 ─────────────────────────────
v2 = pd.read_parquet(GG / "05_산출" / "시나리오_활성화등급_v2.parquet")
act = pd.read_parquet(GG / "02_레이어별" / "침수흔적_활성화강우_격자.parquet")[["grid_id", "act_only_2014"]]
v2 = v2.merge(act, on="grid_id", how="left")
y = v2.trace_flag.values
rc = ((v2.trace_flag == 1) & (v2.n_events >= 2)).values
i14 = ((v2.trace_flag == 1) & (v2.act_only_2014 == 1)).values
T = v2.tier.values
lab = ["T1\n3h 104mm 이하\n(호우경보 문턱)", "T2\n104~142mm", "T3\n142mm 초과\n(2014.8.25급)"]
allc, recc, e14, area = [], [], [], []
for k in [1, 2, 3]:
    m = (T >= 1) & (T <= k)
    allc.append(y[m].sum() / y.sum() * 100)
    recc.append(rc[m].sum() / rc.sum() * 100)
    e14.append(i14[m].sum() / i14.sum() * 100)
    area.append(m.sum() * 0.01)
x = np.arange(3); w = .26
fig, ax = plt.subplots(figsize=(8.4, 4.8))
ax.bar(x - w, recc, w, color=TEAL, label="상습(재발형) 침수")
ax.bar(x, allc, w, color=GREY, label="전체 침수흔적")
ax.bar(x + w, e14, w, color=WARM, label="2014급 단발 침수")
for i in range(3):
    ax.text(i - w, recc[i] + 1.6, f"{recc[i]:.0f}%", ha="center", fontsize=9.5, fontweight="bold", color=TEAL)
    ax.text(i, allc[i] + 1.6, f"{allc[i]:.0f}%", ha="center", fontsize=9.5, color="#555")
    ax.text(i + w, e14[i] + 1.6, f"{e14[i]:.0f}%", ha="center", fontsize=9.5, color=WARM)
    ax.text(i, -9, f"{area[i]:.0f} km²", ha="center", fontsize=9, color="#445", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=9.5)
ax.set_ylabel("누적 포착률 (%)", fontsize=10); ax.set_ylim(0, 112)
ax.legend(fontsize=9.5, frameon=False, ncol=3, loc="upper left")
ax.set_title("강우 강도별 대응구역 확대 — 누적 구조라 포착률은 단조 증가",
             fontsize=12.5, fontweight="bold", pad=10)
clean(ax); fig.tight_layout()
fig.text(.5, -.035, "호우경보 문턱(3h 104mm)에서 상습 침수의 50%, 2014급까지 대비하면 전량 포괄",
         ha="center", fontsize=8.5, color="#555")
fig.savefig(OUT / "F15_강우시나리오_누적포착.png", bbox_inches="tight", dpi=300); plt.close()
print("F15 저장")
print("→", OUT)
