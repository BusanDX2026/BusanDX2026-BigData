# -*- coding: utf-8 -*-
"""V3. 제출용 파이프라인 흐름도 + 표(CSV/MD) 일괄 생성"""
import sys, io, json, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
TEAL, WARM, GREY, INK = "#0d5c58", "#c4703a", "#9aa8a6", "#13201f"

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"
OUT = GG / "06_제출자료"; OUT.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════
# F01. 분석 추진 흐름도
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11.5, 6.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.axis("off")

def box(x, y, w, h, title, body, fc="#ffffff", ec=TEAL, tc=INK, lw=1.4, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.2",
                                facecolor=fc, edgecolor=ec, linewidth=lw))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top",
            fontsize=fs + 1.4, fontweight="bold", color=tc)
    ax.text(x + w / 2, y + h - 6.4, body, ha="center", va="top", fontsize=fs,
            color="#40514f", linespacing=1.55)

def arrow(x1, y1, x2, y2, color=TEAL):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                                 color=color, linewidth=1.6, shrinkA=2, shrinkB=2))

def stage(x, label):
    ax.text(x, 55.4, label, ha="center", fontsize=10.5, fontweight="bold", color=TEAL)
    ax.plot([x - 9.5, x + 9.5], [53.6, 53.6], color=TEAL, lw=2.2, solid_capstyle="butt")

stage(11, "① 데이터 수집")
stage(35, "② 전처리")
stage(59, "③ 모델링")
stage(85, "④ 산출")

box(0.5, 29, 21, 22, "공공데이터 16종",
    "국토지리정보원 DEM 30m\n행안부 침수흔적도·재해위험지구\n환경부 홍수위험지도·토지피복\nSGIS 인구격자 · GIS건물통합\n기상청 AWS/ASOS 시간강우\n부산시 하수맨홀·배수펌프장", fc="#f6faf9")
box(1.5, 6, 19, 20, "Big-데이터웨이브",
    "119 소방출동정보 API\n주요도로 침수 정보\n\n(교차검증에 사용,\n모델 미채택 — 붙임4)", fc="#fbf6f2", ec=WARM)

box(25.5, 30, 19, 20, "격자 정합",
    "EPSG:5186 통일\n100m 격자 81,224개\n인구 dasymetric 배분\nDEM 수문 파생\n(함몰메움 → D8 → HAND)", fc="#f6faf9")
box(25.5, 6, 19, 20, "유발강우 매칭",
    "AWS 16지점 209만 지점-시간\n침수흔적 479폴리곤 전수\n최근접 3지점 IDW\n→ 사건별 최대 1h·3h 강우", fc="#f6faf9")

box(49.5, 30, 19, 20, "M9 침수 감수성",
    "XGBoost 16피처\n자치구 GroupKFold 5-fold\n\nPR-AUC 0.3059\n(무작위 대비 8.6배)", fc="#eaf3f2")
box(49.5, 6, 19, 20, "MS3 활성화 강우",
    "관측 유발강우 회귀\n→ 3단계 등급\n\nT1 ≤104mm/3h\nT2 ≤142 · T3 >142", fc="#eaf3f2")

box(73.5, 18, 25, 26, "선제대응 우선순위",
    "M5 MCDA 결합\n위험 50% + 노출 35% + 대응결핍 15%\n\n격자 81,224 · 행정동 206\n\n상위 10%(81 km²) 지정 시\n침수위험 거주인구 90.4% 포괄",
    fc=TEAL, ec=TEAL, tc="white")
ax.texts[-1].set_color("#dceeec")

arrow(21.5, 40, 25.5, 40); arrow(20.5, 16, 25.5, 16)
arrow(44.5, 40, 49.5, 40); arrow(44.5, 16, 49.5, 16)
arrow(68.5, 40, 73.5, 34)
arrow(68.5, 16, 73.5, 26)

ax.text(50, 1.4, "전 과정 자치구 공간 교차검증 · 위약 대조 · 사건 홀드아웃으로 검증",
        ha="center", fontsize=9.5, color="#5a6b69", style="italic")
fig.savefig(OUT / "F01_추진흐름도.png", bbox_inches="tight", dpi=300, facecolor="white")
plt.close()
print("F01 저장")

# ══════════════════════════════════════════════════════════
# 표 생성
# ══════════════════════════════════════════════════════════
def save(df, name, title):
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    md = [f"### {title}", ""]
    md.append("| " + " | ".join(df.columns) + " |")
    md.append("|" + "|".join(["---"] * len(df.columns)) + "|")
    for _, r in df.iterrows():
        md.append("| " + " | ".join(str(x) for x in r.values) + " |")
    (OUT / f"{name}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"{name} 저장 ({len(df)}행)")

# T01 활용 데이터
save(pd.DataFrame([
    ["국토지리정보원", "수치표고모델 DEM 30m", "2023", "표고·경사·TPI·저지대·HAND·TWI·흐름누적·수계거리", "GeoTIFF"],
    ["행정안전부", "침수흔적도", "2009~2022 (18개 사건)", "침수면적·침수심·발생일시·원인 (타깃)", "SHP"],
    ["행정안전부", "자연재해위험개선지구", "2025", "지정 현황 (기준선 비교)", "API"],
    ["환경부", "홍수위험지도 (하천200년·도시침수100년)", "2023", "범람 면적비·등급", "SHP"],
    ["환경부", "세분류 토지피복지도 11차", "2021", "불투수율·농경·산림·수계·도로 비율", "WFS"],
    ["통계청 SGIS", "인구 격자 (1km) · 행정경계", "2024", "총인구·65세이상·연령별 (dasymetric 배분)", "SHP"],
    ["국토교통부", "GIS건물통합정보", "2024", "건물수·연면적·지하층·최고층수", "SHP"],
    ["기상청", "AWS 방재기상관측 시간자료", "2009~2025 · 14지점", "시간 강수량 (유발강우 산출)", "CSV"],
    ["기상청", "ASOS 종관기상관측 시간자료", "2009~2025 · 2지점", "시간 강수량", "CSV"],
    ["부산광역시", "도시공간정보 하수맨홀", "2025", "준설미실시 비율 (배수 불량 대리지표)", "SHP"],
    ["환경부", "전국 배수펌프장 표준데이터", "2024", "펌프장 거리·개수 (대응역량)", "CSV"],
    ["부산광역시", "지하차도 현황", "2024", "지하차도 개수·연장 (노출)", "CSV"],
    ["부산시 Big-데이터웨이브", "119 소방출동정보", "2020~2025 · 강우일 100일", "배수출동 (독립 교차검증, 모델 미채택)", "API"],
], columns=["제공기관", "데이터명", "시계열", "사용 변수 / 용도", "형식"]), "T01_활용데이터", "활용 데이터 목록")

# T02 성능
d = pd.read_parquet(GG / "05_산출" / "격자_우선순위.parquet").merge(
    pd.read_parquet(GG / "04_모델" / "hazard_score.parquet")[["grid_id", "hazard_oof"]], on="grid_id")
y = d.trace_flag.values; pop = d["pop"].fillna(0).values
tot, par = y.sum(), pop[y == 1].sum()
def tm(sc, p):
    k = int(len(sc) * p)
    j = sc + np.random.RandomState(42).rand(len(sc)) * max(np.ptp(sc), 1e-9) * 1e-9
    m = np.zeros(len(sc), bool); m[np.argpartition(-j, k - 1)[:k]] = True
    return m
rows = []
for p in [0.01, 0.05, 0.10, 0.20, 0.30]:
    mh, mp = tm(d.hazard_oof.values, p), tm(d.priority.values, p)
    rows.append([f"{p:.0%}", f"{int(len(d)*p):,}", f"{int(len(d)*p)*0.01:.0f} km²",
                 f"{y[mh].sum()/tot:.1%}", f"{y[mh].mean():.1%}", f"{(y[mh].sum()/tot)/p:.1f}배",
                 f"{pop[mp & (y==1)].sum()/par:.1%}", f"{pop[mp].sum():,.0f}명"])
save(pd.DataFrame(rows, columns=["지정 규모", "격자수", "면적", "침수흔적 포착(M9)", "정밀도(M9)",
                                 "리프트(M9)", "위험인구 포착(M5)", "총 커버인구(M5)"]),
     "T02_성능요약", "지정 규모별 성능 요약")

# T03 검증
save(pd.DataFrame([
    ["공간 교차검증", "자치구 GroupKFold 5-fold", "PR-AUC 0.3059", "랜덤 K-fold(0.59)는 +141% 과대평가 — 공간 자기상관 제거"],
    ["사건 홀드아웃", "2020 대형호우 학습 제외", "리프트 16.8배 · 상위10% 68.0% 포착", "평범한 대형호우는 잘 일반화"],
    ["사건 홀드아웃", "2011-07-27 학습 제외", "리프트 40.9배", "동일"],
    ["사건 홀드아웃", "2014-08-25 학습 제외", "리프트 3.6배 · 상위10% 33.8%", "미증유 극한은 평상시 안전지까지 침수 — 전면대응 국면"],
    ["기준선 비교", "행정 재해위험지구 지정", "12.1% (본 모델 59.1%)", "현행 방식 대비 4.9배"],
    ["기준선 비교", "무작위 / MCDA 동일가중", "10.2% / 7.7%", "물리지표 단순 평균은 무작위보다 나쁨"],
    ["편향 통제", "자치구 내부 순위상관", "Spearman 중앙값 +0.27", "조사편향과 무관하게 구 내부 순위는 유효"],
    ["위약 대조", "행정동 난수 5개 투입", "PR-AUC −28.9%", "SHAP 기여율 높은 피처 4/5를 '행정동 지문'으로 기각"],
    ["독립 출처", "119 배수출동 (인구 통제)", "M9 ρ=+0.201 (p=0.040)", "다른 출처에서도 물리 위험도 신호 유의"],
], columns=["검증 유형", "방법", "결과", "해석"]), "T03_검증요약", "검증 수행 결과")

# T04 행정동 TOP 20
dg = pd.read_csv(GG / "05_산출" / "행정동_우선순위.csv")
cp = "risk_pop" if "risk_pop" in dg.columns else [c for c in dg.columns if "위험" in c][0]
cn = "adm_nm" if "adm_nm" in dg.columns else "행정동"
cs = "sgg_nm" if "sgg_nm" in dg.columns else "자치구"
ct = "top10_ratio" if "top10_ratio" in dg.columns else None
cf = "flood_grids" if "flood_grids" in dg.columns else None
t = dg.nlargest(20, cp).reset_index(drop=True)
out = pd.DataFrame({
    "순위": range(1, len(t) + 1), "자치구": t[cs], "행정동": t[cn],
    "위험 노출인구": t[cp].round(0).astype(int).map("{:,}".format),
    "상위10% 격자비율": (t[ct] * 100).round(1).astype(str) + "%" if ct else "-",
    "침수흔적 격자": t[cf].astype(int) if cf else "-"})
save(out, "T04_행정동_우선순위TOP20", "선제대응 우선 행정동 TOP 20")

# T05 한계
save(pd.DataFrame([
    ["타깃 편향", "침수흔적도는 '기록된 침수' — 기장군이 전체의 31%, 인구는 5%", "인구·펌프를 학습에서 제외, 자치구 내부 순위로 검증", "119 출동·풍수해보험 등 독립 타깃 확보"],
    ["단일 사건 지배", "침수흔적의 57%가 2014-08-25 하나", "사건 홀드아웃으로 일반화 별도 측정 (3.6배)", "119 출동 로그로 사건 수 확대"],
    ["활성화 강우 정밀도", "회귀 R² 0.242 · 분리 AUC 0.688 — 등급은 광역 밴드", "격자별 임계값 단정 금지, 상대 순서로만 사용", "레이더 격자강수(RN1)로 사건 내 공간차 확보"],
    ["강서 삼각주", "광대한 저지대이나 농경지라 침수기록 없음 → 역방향 학습", "해당 자치구는 물리 MCDA 점수로 대체 적용", "도시침수지도 전역 확대 시 해소"],
    ["하수관망 부재", "내수침수의 1순위 결측 변수, 부산시 미공개(지하시설물 보안)", "하수맨홀 준설미실시 비율로 부분 대리", "본선 진출 시 부산시 하수도과 공식 요청"],
    ["정밀도 한계", "상위 10% 정밀도 21.0% — 79%가 오탐", "'침수 예측기'가 아닌 '점검 우선순위 도구'로 위치 규정", "하수관망 확보 시 개선 여지"],
], columns=["한계", "내용", "현재 대응", "보완 계획"]), "T05_한계와보완", "결과의 한계 및 보완점")
print("→", OUT)
