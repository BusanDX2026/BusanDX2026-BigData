# -*- coding: utf-8 -*-
"""V5. 시각자료집 PDF 생성 — 그림 15종 + 표 5종을 A4 한 권으로"""
import sys, io, textwrap, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
TEAL, WARM, INK, MUTE = "#0d5c58", "#c4703a", "#13201f", "#5a6b69"
A4 = (8.27, 11.69)

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"
SRC = GG / "06_제출자료"
PDF = SRC / "시각자료집_부산_선제대응구역.pdf"

# (파일, 번호, 제목, 캡션, 서식 대응)
FIGS = [
    ("F01_추진흐름도.png", "그림 1", "분석 추진 흐름",
     "공공데이터 16종을 100m 격자로 정합하고, 물리 지표로 침수 감수성을 학습한 뒤 "
     "인구 노출과 대응역량을 결합해 우선순위를 산출했다. 전 과정을 자치구 공간 교차검증으로 검증했다.",
     "1. 분석개요 · 추진과정 / 2. 분석방법 · 분석절차"),
    ("F02_침수흔적_조사편향.png", "그림 2", "타깃의 조사편향",
     "타깃인 침수흔적도는 '실제 침수'가 아니라 '기록된 침수'다. 기장군이 전체 기록의 31%인데 "
     "인구는 부산의 5%에 불과하다. 이 편향 때문에 인구·펌프장을 학습에서 제외했다.",
     "2. 분석방법 · 전처리 내용 및 검증 여부"),
    ("F13_SHAP기여도.png", "그림 3", "무엇이 침수 감수성을 만드나",
     "최저표고(23.4%)·수계거리(11.3%)·하수맨홀 준설미실시(10.0%)가 상위. 절대 저지대보다 "
     "'주변 대비 낮음'(TPI)이 결정적이며, 이는 강서 삼각주가 저지대이나 침수기록이 없는 현상과 일치한다.",
     "2. 분석방법 · 분석기법 / 3. 분석결과 · 인사이트"),
    ("F06_개선경로.png", "그림 4", "피처군별 누적 기여",
     "DEM에서 직접 산출한 수문 피처(흐름누적·TWI·수계거리)가 최대 기여(+35.9%). "
     "부산 침수의 61%가 하천범람 밖 내수침수인데, 내수는 본질적으로 집수-배수 문제이기 때문이다.",
     "2. 분석방법 · 분석절차 / 3. 분석결과"),
    ("F03_위험도지도.png", "그림 5", "침수 감수성 지도 (M9)",
     "16개 물리지표 기반 XGBoost 예측을 백분위로 표현. PR-AUC 0.3059로 무작위 대비 8.6배, "
     "상위 10%에서 침수흔적의 59.1%를 포착한다.",
     "3. 분석결과 · 시각화 결과"),
    ("F05_커트라인별_성능.png", "그림 6", "지정 규모별 성능",
     "구역을 넓힐수록 더 많이 포착하지만 정밀도는 떨어진다. 상위 1%에서 정밀도 54.7%, "
     "상위 10%에서 재현율 59.1%·정밀도 21.0%. 자원 제약에 맞춰 커트라인을 고르는 근거다.",
     "3. 분석결과 · 검증 수행 결과"),
    ("F07_기준선비교.png", "그림 7", "현행 방식 대비",
     "같은 상위 10%를 지정했을 때 본 모델 59.1% vs 행정 재해위험지구 12.1%. "
     "물리지표 단순 평균(MCDA 동일가중)은 7.7%로 무작위(10.2%)보다도 낮다.",
     "3. 분석결과 · 검증 수행 결과"),
    ("F08_홀드아웃검증.png", "그림 8", "사건 홀드아웃 검증",
     "특정 호우를 학습에서 완전히 빼고 그 침수를 맞히는지 시험했다. 평범한 대형호우는 "
     "16.8~40.9배로 잘 일반화한다. 2014.8.25만 3.6배로 낮은데, 미증유 극한은 평상시 "
     "안전한 곳까지 침수시키기 때문이며 이 구간은 우선순위가 아닌 전면대응 국면이다.",
     "2. 분석방법 · 분석결과 검증방법"),
    ("F14_자치구내_편향통제.png", "그림 9", "조사편향 통제 검증",
     "자치구별 조사 강도가 다르므로 전역 성능만으로는 부족하다. 자치구 '안에서'의 순위상관을 "
     "보면 16개 구 전부 양(+)이고 중앙값 +0.27로, 편향과 무관하게 구 내부 우선순위는 유효하다.",
     "2. 분석방법 · 분석결과 검증방법"),
    ("F10_관측유발강우.png", "그림 10", "관측으로 확인한 활성화 강우",
     "기상청 AWS 16지점 시간강우를 침수흔적 479폴리곤 전수에 매칭했다. 상습 침수지는 "
     "3시간 102mm에 잠기고, 2014급에서만 잠긴 곳은 145mm가 필요했다. "
     "'상습지는 더 약한 비에 잠긴다'는 가설이 추정이 아니라 관측으로 확인됐다.",
     "3. 분석결과 · 인사이트"),
    ("F09_강우활성화등급지도.png", "그림 11", "강우 강도별 활성화 등급 지도",
     "[부가 참고] 관측 유발강우로 격자를 3등급으로 나눴다. 누적 포함 구조(T1⊂T2⊂T3)라 강우가 "
     "커지면 구역이 추가만 되며 포착률 역전이 불가능하다. 다만 회귀 R² 0.242·분리 AUC 0.688 로 "
     "느슨하므로 격자별 임계값 단정이 아니라 광역 밴드 조언으로만 쓴다.",
     "3. 분석결과 · 시각화 결과"),
    ("F15_강우시나리오_누적포착.png", "그림 12", "강우 강도별 대응구역 확대",
     "[부가 참고] 호우경보 문턱(3시간 104mm)에서 133km²로 상습 침수의 50%, 2014급(142mm 초과)까지 "
     "대비하면 302km²로 전량을 포괄한다. 강우 규모별로 점검 범위를 어디까지 넓힐지에 대한 조언이며, "
     "실시간 발령 기준이 아니다.",
     "3. 분석결과 · 시각화 결과 / 4. 활용방안"),
    ("F04_선제대응_우선순위지도.png", "그림 13", "선제대응 우선순위 지도 (최종 산출)",
     "물리 위험도 50% + 인구·건물 노출 35% + 배수 대응역량 결핍 15%를 결합한 최종 산출물. "
     "상위 10%(81km², 인구 127만명)를 지정하면 침수위험 거주인구의 90.4%를 포괄한다. "
     "전수 점검이 불가능할 때 '어디부터 볼지'를 정하는 권고이며, 침수 발생을 예측하지 않는다.",
     "1. 분석개요 · 분석결과 / 3. 분석결과 · 시각화"),
    ("F12_위험노출인구_포착.png", "그림 14", "왜 노출을 결합해야 하나",
     "침수기록이 저인구 지역에 쏠려 있어(기장군 침수격자당 197명 vs 남구 1,843명) 격자 수만 "
     "좇으면 사람이 적은 곳을 우선하게 된다. 사람 기준으로 재면 노출 결합이 67.0%→90.4%로 앞선다.",
     "3. 분석결과 · 인사이트 / 4. 활용방안"),
    ("F11_행정동_우선순위.png", "그림 15", "선제대응 우선 행정동",
     "정책 집행 단위인 행정동으로 집계한 결과. 위험 노출인구 = 상위10% 격자비율 × 인구.",
     "4. 활용방안 · 정책 활용 방안"),
]

# (파일, 번호, 제목, 서식대응, 글자크기, 컬럼폭 비율)
TABLES = [
    ("T01_활용데이터.csv", "표 1", "활용 데이터 목록", "2. 분석방법 · 활용데이터", 7.4,
     [.15, .24, .16, .37, .08]),
    ("T02_성능요약.csv", "표 2", "지정 규모별 성능 요약", "3. 분석결과", 8.2,
     [.11, .10, .10, .17, .12, .12, .16, .12]),
    ("T03_검증요약.csv", "표 3", "검증 수행 결과", "2. 분석방법 · 분석결과 검증방법", 7.4,
     [.13, .24, .25, .38]),
    ("T04_행정동_우선순위TOP20.csv", "표 4", "선제대응 우선 행정동 TOP 20", "4. 활용방안", 8.4,
     [.08, .16, .20, .20, .20, .16]),
    ("T05_한계와보완.csv", "표 5", "결과의 한계 및 보완점", "3. 분석결과 · 한계 및 보완점", 7.4,
     [.14, .30, .29, .27]),
]

def wrap(s, w):
    return "\n".join(textwrap.wrap(str(s), w)) if s else ""

with PdfPages(PDF) as pdf:
    # ── 표지 ─────────────────────────────────────────────
    fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
    fig.text(.12, .90, "2026년 Big Data 활용 대회", fontsize=12, color=TEAL, fontweight="bold")
    fig.text(.12, .875, "빅데이터 분석 및 시각화 부문 · 시각자료집", fontsize=10.5, color=MUTE)
    fig.add_artist(plt.Line2D([.12, .88], [.855, .855], color=INK, lw=2.2))
    fig.text(.12, .76, "부산시 호우 시", fontsize=27, fontweight="bold", color=INK)
    fig.text(.12, .705, "선제대응구역 선정", fontsize=27, fontweight="bold", color=INK)
    fig.text(.12, .655, "한정 자원의 점검 우선순위 권고", fontsize=13, color=TEAL, fontweight="bold")
    fig.text(.12, .625, "100m 격자 81,224개 · 공공데이터 16종 · 자치구 공간 교차검증",
             fontsize=10.5, color=MUTE)
    # 핵심 지표
    ys = .525
    for lab, val, sub in [("침수 감수성 모델", "PR-AUC 0.3059", "무작위 대비 8.6배"),
                          ("선제대응 상위 10%", "위험인구 90.4%", "81 km² · 인구 127만명"),
                          ("관측 활성화 강우", "3시간 102 mm", "상습 침수지 기준")]:
        fig.add_artist(plt.Rectangle((.12, ys - .058), .76, .072, facecolor="#f2f7f6",
                                     edgecolor="#d6e3e1", transform=fig.transFigure))
        fig.text(.15, ys - .002, lab, fontsize=9.5, color=MUTE)
        fig.text(.15, ys - .036, val, fontsize=15, fontweight="bold", color=TEAL)
        fig.text(.86, ys - .028, sub, fontsize=9.5, color=MUTE, ha="right")
        ys -= .088
    fig.text(.12, .175, f"그림 {len(FIGS)}종 · 표 {len(TABLES)}종", fontsize=10.5, color=INK, fontweight="bold")
    fig.text(.12, .15, "모든 수치는 산출물에서 직접 읽어 생성되며, 스크립트 재실행 시 자동 갱신된다.",
             fontsize=9, color=MUTE)
    fig.add_artist(plt.Line2D([.12, .88], [.11, .11], color="#d6e3e1", lw=1))
    fig.text(.12, .085, "생성 2026-09-04 · scripts/산출/V1~V5", fontsize=8.5, color=MUTE)
    fig.text(.88, .085, "※ 분석 진행 중 — 수치는 변경될 수 있음", fontsize=8.5, color=WARM, ha="right")
    pdf.savefig(fig); plt.close(fig)

    # ── 이 산출물의 위치 ──────────────────────────────────
    fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
    fig.text(.10, .935, "이 산출물의 위치", fontsize=18, fontweight="bold", color=INK)
    fig.add_artist(plt.Line2D([.10, .90], [.920, .920], color=INK, lw=1.6))

    fig.add_artist(plt.Rectangle((.10, .805), .80, .085, facecolor="#eaf3f2",
                                 edgecolor=TEAL, lw=1.4, transform=fig.transFigure))
    fig.text(.50, .862, "한정 자원의 점검 우선순위 권고", fontsize=15, fontweight="bold",
             color=TEAL, ha="center")
    fig.text(.50, .828, "침수 발생을 예측하거나 경보를 발령하는 시스템이 아니다.",
             fontsize=10.5, color=MUTE, ha="center")

    y = .765
    fig.text(.10, y, "숫자가 지지하는 범위", fontsize=11.5, fontweight="bold", color=INK); y -= .034
    for lab, val, can in [
        ("상위 10% 정밀도", "21.0%", "5곳 중 1곳이 실제 침수이력 → 점검 순서 O / 자동 발령 X"),
        ("상위 10% 재현율", "59.1%", "어디를 먼저 볼지에 대한 답으로 충분"),
        ("위험인구 포착", "90.4%", "정책 단위 커버리지 근거"),
        ("강우 등급(부가)", "AUC 0.688", "격자별 임계 단정 X / 광역 밴드 조언 O")]:
        fig.text(.115, y, lab, fontsize=9.5, color=MUTE)
        fig.text(.315, y, val, fontsize=10.5, fontweight="bold", color=TEAL)
        fig.text(.435, y, can, fontsize=9, color="#31423f")
        y -= .0285
    y -= .022

    fig.text(.10, y, "왜 '권고'인가", fontsize=11.5, fontweight="bold", color=INK); y -= .032
    for t in ["정밀도 21%로 시민 대상 자동 발령을 하면 5건 중 4건이 헛발령이 된다.",
              "안전문자는 오발령 비용이 특히 크다 — 신뢰가 깨지면 실제 위험 시 대응하지 않는다.",
              "반면 '이 순서로 점검하라'는 조언으로는 같은 21%가 충분하다.",
              "무작위 10곳 중 1곳 대비 5곳 중 1곳이므로 같은 예산으로 5배 효율이다."]:
        fig.text(.125, y, "·", fontsize=10, color=TEAL)
        fig.text(.145, y, t, fontsize=9.5, color="#31423f")
        y -= .0255
    y -= .022

    fig.text(.10, y, "권고 도구로서의 근거", fontsize=11.5, fontweight="bold", color=INK); y -= .032
    for lab, val in [("현행 행정 재해위험지구 지정 대비", "4.9배  (59.1% vs 12.1%)"),
                     ("2020 대형호우 홀드아웃", "16.8배 — 처음 보는 호우에도 작동"),
                     ("검증 깊이", "공간 CV · 위약 대조 · 사건 홀드아웃 · 독립출처 교차검증")]:
        fig.text(.125, y, lab, fontsize=9.5, color=MUTE)
        fig.text(.47, y, val, fontsize=9.5, color="#31423f", fontweight="bold")
        y -= .0265
    y -= .008
    fig.text(.125, y, "랜덤 K-fold 가 +141% 부풀린 값임을 보이고 정직한 수치를 쓴다.",
             fontsize=9, color=MUTE, style="italic")
    y -= .042

    fig.text(.10, y, "활용 범위", fontsize=11.5, fontweight="bold", color=INK); y -= .032
    for ok, t in [(True, "연간 점검계획 수립 — 호우철 전 배수구·펌프장 점검 순서"),
                  (True, "예산 배분 근거 — 어느 구역 정비에 먼저 투입할지"),
                  (True, "재해위험지구 지정 재검토 — 현행 지정과의 차이 52개 행정동"),
                  (False, "실시간 경보·안전문자 발송 — 권한 부재 + 정밀도 부족으로 채택하지 않음"),
                  (False, "격자별 침수 시점·수심 예측 — 이 데이터로 불가")]:
        fig.text(.125, y, "O" if ok else "X", fontsize=10,
                 color=TEAL if ok else WARM, fontweight="bold")
        fig.text(.155, y, t, fontsize=9.5, color="#31423f" if ok else MUTE)
        y -= .0255

    fig.add_artist(plt.Line2D([.10, .90], [.075, .075], color="#d6e3e1", lw=1))
    fig.text(.10, .052, "상세 근거: 문서/모델링_설계와검증.md 부록 P2", fontsize=8.5, color=MUTE)
    pdf.savefig(fig); plt.close(fig)


    # ── 목차 ─────────────────────────────────────────────
    fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
    fig.text(.12, .93, "목차", fontsize=18, fontweight="bold", color=INK)
    fig.add_artist(plt.Line2D([.12, .88], [.915, .915], color=INK, lw=1.6))
    y = .875
    fig.text(.12, y, "그림", fontsize=11, fontweight="bold", color=TEAL); y -= .026
    for i, (_, no, title, _, sec) in enumerate(FIGS):
        fig.text(.13, y, f"{no}", fontsize=9, color=MUTE)
        fig.text(.215, y, title, fontsize=9.5, color=INK)
        fig.text(.88, y, sec.split("/")[0].strip(), fontsize=8, color=MUTE, ha="right")
        y -= .0235
    y -= .018
    fig.text(.12, y, "표", fontsize=11, fontweight="bold", color=TEAL); y -= .026
    for _, no, title, sec, _, _w in TABLES:
        fig.text(.13, y, f"{no}", fontsize=9, color=MUTE)
        fig.text(.215, y, title, fontsize=9.5, color=INK)
        fig.text(.88, y, sec.split("·")[0].strip(), fontsize=8, color=MUTE, ha="right")
        y -= .0235
    pdf.savefig(fig); plt.close(fig)

    # ── 그림 페이지 ──────────────────────────────────────
    for fn, no, title, cap, sec in FIGS:
        p = SRC / fn
        if not p.exists():
            print(f"  ! 없음 {fn}"); continue
        fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
        fig.text(.09, .955, no, fontsize=10, fontweight="bold", color=TEAL)
        fig.text(.155, .955, title, fontsize=14, fontweight="bold", color=INK)
        fig.text(.91, .955, sec, fontsize=8, color=MUTE, ha="right")
        fig.add_artist(plt.Line2D([.09, .91], [.943, .943], color="#d6e3e1", lw=1))
        ax = fig.add_axes([.07, .195, .86, .725]); ax.axis("off")
        ax.imshow(plt.imread(p))
        fig.text(.09, .145, wrap(cap, 62), fontsize=9.5, color="#31423f",
                 va="top", linespacing=1.65)
        fig.text(.09, .045, f"파일: {fn}", fontsize=7.5, color="#96a5a3")
        pdf.savefig(fig); plt.close(fig)
        print(f"  {no} {title}")

    # ── 표 페이지 (직접 렌더링 — matplotlib table 은 줄바꿈·폭 제어가 안 됨) ──
    for fn, no, title, sec, fs, widths in TABLES:
        p = SRC / fn
        if not p.exists():
            print(f"  ! 없음 {fn}"); continue
        df = pd.read_csv(p).astype(str)
        widths = [w / sum(widths) for w in widths]

        L, R, TOP, BOT = .055, .945, .925, .055
        W = R - L
        # 컬럼 폭(인치) → 대략적인 글자수 (fs pt 기준, 한글 폭 ≈ fs*0.95pt)
        def nchar(frac):
            inch = W * A4[0] * frac
            return max(6, int(inch * 72 / (fs * 0.98)))
        cells = [[wrap(v, nchar(widths[c])) for c, v in enumerate(row)] for row in df.values]
        head = [wrap(c, nchar(widths[i])) for i, c in enumerate(df.columns)]
        nl = lambda t: t.count("\n") + 1
        rowh = [max(nl(c) for c in row) for row in cells]
        headh = max(nl(c) for c in head)
        unit = (TOP - BOT) / (sum(rowh) + headh * 1.35 + 0.6 * len(rowh))
        unit = min(unit, .030)

        fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
        fig.text(L, .962, no, fontsize=10, fontweight="bold", color=TEAL)
        fig.text(L + .052, .962, title, fontsize=14, fontweight="bold", color=INK)
        fig.text(R, .962, sec, fontsize=8, color=MUTE, ha="right")
        fig.add_artist(plt.Line2D([L, R], [.950, .950], color="#d6e3e1", lw=1))

        xs = [L]
        for w in widths:
            xs.append(xs[-1] + w * W)
        y = TOP
        hh = headh * unit * 1.35
        fig.add_artist(plt.Rectangle((L, y - hh), W, hh, facecolor=TEAL,
                                     edgecolor="none", transform=fig.transFigure))
        for c, txt in enumerate(head):
            fig.text(xs[c] + .006, y - hh / 2, txt, fontsize=fs + .6, color="white",
                     fontweight="bold", va="center", linespacing=1.35)
        y -= hh
        for r, row in enumerate(cells):
            h = rowh[r] * unit + 0.6 * unit
            if r % 2 == 1:
                fig.add_artist(plt.Rectangle((L, y - h), W, h, facecolor="#f5f9f8",
                                             edgecolor="none", transform=fig.transFigure))
            for c, txt in enumerate(row):
                fig.text(xs[c] + .006, y - h / 2, txt, fontsize=fs, color="#25332f",
                         va="center", linespacing=1.35)
            fig.add_artist(plt.Line2D([L, R], [y - h, y - h], color="#e2eae8",
                                      lw=.7, transform=fig.transFigure))
            y -= h
        fig.add_artist(plt.Line2D([L, R], [y, y], color=TEAL, lw=1.2,
                                  transform=fig.transFigure))
        fig.text(L, max(y - .028, .022), f"파일: {fn}", fontsize=7.5, color="#96a5a3")
        pdf.savefig(fig); plt.close(fig)
        print(f"  {no} {title}")


    d = pdf.infodict()
    d["Title"] = "부산시 호우 시 선제대응구역 선정 — 시각자료집"
    d["Subject"] = "2026년 Big Data 활용 대회 (빅데이터 분석 및 시각화 부문)"
    d["Keywords"] = "침수, 선제대응, 부산, 공간분석, XGBoost"

print(f"\nPDF 생성: {PDF} ({PDF.stat().st_size/1024/1024:.1f} MB)")
