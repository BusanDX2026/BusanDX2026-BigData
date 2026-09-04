# -*- coding: utf-8 -*-
"""V1. 제출용 지도 4종 — 위험도 / 선제대응 우선순위 / 강우 활성화 등급 / 침수흔적·편향"""
import sys, io, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"
OUT = GG / "06_제출자료"; OUT.mkdir(parents=True, exist_ok=True)

pr = pd.read_parquet(GG / "05_산출" / "격자_우선순위.parquet")
hz = pd.read_parquet(GG / "04_모델" / "hazard_score.parquet")[["grid_id", "hazard_oof"]]
d = pr.merge(hz, on="grid_id", how="left")
v2 = pd.read_parquet(GG / "05_산출" / "시나리오_활성화등급_v2.parquet")[["grid_id", "tier"]]
d = d.merge(v2, on="grid_id", how="left")
sgg = gpd.read_file(GG / "01_격자" / "부산_시군구.gpkg").to_crs(5186)

# grid_id "3607_2703" → 열/행 인덱스
gx = d.grid_id.str.split("_").str[0].astype(int)
gy = d.grid_id.str.split("_").str[1].astype(int)
c0, r0 = gx.min(), gy.min()
NC, NR = gx.max() - c0 + 1, gy.max() - r0 + 1
col = (gx - c0).values
row = (gy.max() - gy).values          # 위가 북쪽
EXT = [c0 * 100, (gx.max() + 1) * 100, gy.min() * 100, (gy.max() + 1) * 100]
print(f"래스터 {NR} x {NC} · 격자 {len(d):,}")

def canvas(vals, fill=np.nan):
    a = np.full((NR, NC), fill, dtype=float)
    a[row, col] = vals
    return a

def draw(ax, arr, cmap, norm=None, vmin=None, vmax=None):
    ax.imshow(arr, extent=EXT, origin="upper", cmap=cmap, norm=norm,
              vmin=vmin, vmax=vmax, interpolation="nearest")
    sgg.boundary.plot(ax=ax, color="white", linewidth=0.6, alpha=0.85)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

def scalebar(ax, km=5):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    bx = x0 + (x1 - x0) * 0.06; by = y0 + (y1 - y0) * 0.06
    ax.plot([bx, bx + km * 1000], [by, by], color="#111", lw=2.5, solid_capstyle="butt")
    ax.text(bx + km * 500, by + (y1 - y0) * 0.012, f"{km} km", ha="center", va="bottom",
            fontsize=8, color="#111")

# ─────────────────────────────────────────────────────────
# F03. M9 침수 감수성
# ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.4, 8.2))
draw(ax, canvas(d.hazard_oof.rank(pct=True).values), "YlOrRd", vmin=0, vmax=1)
ax.set_title("침수 감수성 (M9)\n16개 물리지표 기반 · 자치구 공간 교차검증",
             fontsize=13, fontweight="bold", pad=12)
sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(0, 1))
cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("위험도 백분위", fontsize=9); cb.ax.tick_params(labelsize=8)
scalebar(ax)
fig.text(0.5, 0.045, "PR-AUC 0.3059 (무작위 대비 8.6배) · 상위 10%에서 침수흔적의 59.1% 포착",
         ha="center", fontsize=8.5, color="#444")
fig.savefig(OUT / "F03_위험도지도.png", bbox_inches="tight", dpi=300); plt.close(fig)
print("F03 저장")

# ─────────────────────────────────────────────────────────
# F04. M5 선제대응 우선순위 (등급 구간)
# ─────────────────────────────────────────────────────────
p = d.priority.rank(pct=True, ascending=False)
grade = np.select([p <= .05, p <= .10, p <= .20, p <= .30], [4, 3, 2, 1], 0).astype(float)
grade[d.priority.isna().values] = np.nan
cmap = ListedColormap(["#eef2f1", "#cfe3e0", "#7bbdb5", "#2b8c83", "#0d5c58"])
fig, ax = plt.subplots(figsize=(7.4, 8.2))
draw(ax, canvas(grade), cmap, norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5, 4.5], 5))
ax.set_title("선제대응 우선순위 (M5 MCDA)\n위험 50% + 노출 35% + 대응결핍 15%",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(handles=[Patch(facecolor="#0d5c58", label="1순위  상위 5%"),
                   Patch(facecolor="#2b8c83", label="2순위  5~10%"),
                   Patch(facecolor="#7bbdb5", label="3순위  10~20%"),
                   Patch(facecolor="#cfe3e0", label="4순위  20~30%"),
                   Patch(facecolor="#eef2f1", label="대상 외")],
          loc="upper right", fontsize=8.5, frameon=True, framealpha=.95)
scalebar(ax)
fig.text(0.5, 0.045, "상위 10%(81 km²) 지정 시 침수위험 거주인구 27.9만명 중 90.4% 포괄",
         ha="center", fontsize=8.5, color="#444")
fig.savefig(OUT / "F04_선제대응_우선순위지도.png", bbox_inches="tight", dpi=300); plt.close(fig)
print("F04 저장")

# ─────────────────────────────────────────────────────────
# F09. 강우 활성화 등급 (MS3)
# ─────────────────────────────────────────────────────────
t = d.tier.fillna(0).values.astype(float)
cmap2 = ListedColormap(["#eef2f1", "#c94f3a", "#e0955a", "#f0d59a"])
fig, ax = plt.subplots(figsize=(7.4, 8.2))
draw(ax, canvas(t), cmap2, norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4))
ax.set_title("강우 강도별 활성화 등급 (MS3)\n관측 유발강우 기반 · 누적 포함 구조",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(handles=[Patch(facecolor="#c94f3a", label="T1 상시취약   3h 104mm 이하"),
                   Patch(facecolor="#e0955a", label="T2 집중호우   104~142mm"),
                   Patch(facecolor="#f0d59a", label="T3 극한호우   142mm 초과"),
                   Patch(facecolor="#eef2f1", label="대상 외")],
          loc="upper right", fontsize=8.5, frameon=True, framealpha=.95)
scalebar(ax)
fig.text(0.5, 0.045, "관측: 상습 침수지 유발강우 3시간 102mm · 2014급 단발지 145mm",
         ha="center", fontsize=8.5, color="#444")
fig.savefig(OUT / "F09_강우활성화등급지도.png", bbox_inches="tight", dpi=300); plt.close(fig)
print("F09 저장")

# ─────────────────────────────────────────────────────────
# F02. 침수흔적 분포 + 조사편향
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.6), gridspec_kw={"width_ratios": [1, 1.05]})
tr = d.trace_flag.values.astype(float); tr[tr == 0] = np.nan
draw(axes[0], canvas(np.full(len(d), 0.12)), ListedColormap(["#e9edec"]), vmin=0, vmax=1)
axes[0].imshow(canvas(tr), extent=EXT, origin="upper",
               cmap=ListedColormap(["#1f4e8c"]), interpolation="nearest")
sgg.boundary.plot(ax=axes[0], color="white", linewidth=0.6)
axes[0].set_xticks([]); axes[0].set_yticks([])
for s in axes[0].spines.values(): s.set_visible(False)
axes[0].set_title("침수흔적 분포 (2009~2022, 2,893격자)", fontsize=12, fontweight="bold")
scalebar(axes[0])

b = d.groupby("sgg_nm").agg(흔적=("trace_flag", "sum"), 인구=("pop", "sum")).reset_index()
b["격자당인구"] = b.인구 / b.흔적.replace(0, np.nan)
b = b.sort_values("흔적", ascending=True)
ax2 = axes[1]
ax2.barh(b.sgg_nm, b.흔적, color="#1f4e8c", height=.68)
ax2.set_xlabel("침수흔적 격자 수", fontsize=10)
ax2.set_title("자치구별 침수흔적 — 조사편향의 근거", fontsize=12, fontweight="bold")
ax2.tick_params(labelsize=9); ax2.grid(axis="x", alpha=.25, lw=.6)
for sp in ["top", "right"]: ax2.spines[sp].set_visible(False)
for i, (n, v) in enumerate(zip(b.sgg_nm, b.흔적)):
    ax2.text(v + 8, i, f"{int(v):,}", va="center", fontsize=8, color="#333")
ax2.annotate("기장군 886격자 = 전체의 31%\n(인구는 부산의 5%)",
             xy=(886, len(b) - 1), xytext=(560, len(b) - 4.6), fontsize=9, color="#a03c1e",
             arrowprops=dict(arrowstyle="->", color="#a03c1e", lw=1.2))
fig.suptitle("타깃은 '실제 침수'가 아니라 '기록된 침수'다", fontsize=13.5, fontweight="bold", y=.985)
fig.tight_layout()
fig.savefig(OUT / "F02_침수흔적_조사편향.png", bbox_inches="tight", dpi=300); plt.close(fig)
print("F02 저장")
print("→", OUT)
