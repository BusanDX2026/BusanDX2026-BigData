from pathlib import Path
import csv
import json
import subprocess
from collections import defaultdict

from PIL import Image
from pypdf import PdfReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "output/활용데이터_CSV/활용데이터_CSV"
OUT = ROOT / "output/pdf/수문장_데이터시각화_3D유지_시안.pdf"
TMP = ROOT / "tmp/pdfs/3D유지_시안"
TMP.mkdir(parents=True, exist_ok=True)
OUT.parent.mkdir(parents=True, exist_ok=True)

SOURCE_PDF = Path.home() / "Downloads/수문장_데이터시각화.pdf"
SOURCE_PAGE = TMP / "기존시각화_300dpi.png"
VISUAL_3D = TMP / "3D_핵심시각물.png"
if not SOURCE_PDF.exists():
    raise FileNotFoundError(f"3D 시각물 원본을 찾을 수 없습니다: {SOURCE_PDF}")
subprocess.run(
    ["pdftoppm", "-png", "-r", "300", "-f", "1", "-singlefile", str(SOURCE_PDF), str(SOURCE_PAGE.with_suffix(""))],
    check=True,
)
with Image.open(SOURCE_PAGE) as source_page:
    # 기존 자료의 핵심인 3D 격자와 평면지도·순위표·범례만 보존한다.
    source_page.crop((100, 680, 2010, 2050)).save(VISUAL_3D)


def read_csv(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


m5 = read_csv("05_M5_최종점검순위_100m격자.csv")
m9 = read_csv("04_M9_침수감수성_100m격자.csv")
dong_ranking = read_csv("06_M5_행정동별_점검순위.csv")
validation = json.loads(
    (ROOT / "공공데이터/가공데이터/04_모델/M9_검증.json").read_text(encoding="utf-8")
)
comparison = next(row for row in validation["hazdist"] if row["flag"] == "hazdist_flood")

labels = {row["grid_id"]: int(row["trace_flag"]) for row in m5}
oof_order = sorted(m9, key=lambda row: float(row["hazard_oof"]), reverse=True)
positives = sum(labels.values())
top10_k = int(len(m9) * 0.10)
top20_k = int(len(m9) * 0.20)
top10_hits = sum(labels[row["grid_id"]] for row in oof_order[:top10_k])
top20_hits = sum(labels[row["grid_id"]] for row in oof_order[:top20_k])
top10_precision = top10_hits / top10_k
top10_capture = top10_hits / positives
top20_capture = top20_hits / positives
matched_hits = sum(labels[row["grid_id"]] for row in oof_order[: comparison["k"]])
matched_precision = matched_hits / comparison["k"]

assert len(m5) == 81224 and positives == 2893
assert top10_k == 8122 and top10_hits == 1709
assert top20_k == 16244 and top20_hits == 2180
assert round(top10_precision * 100, 1) == 21.0
assert round(top10_capture * 100, 1) == 59.1
assert round(top20_capture * 100, 1) == 75.4
assert comparison["k"] == 772
assert round(comparison["base"] * 100, 1) == 19.3
assert round(matched_precision * 100, 1) == 55.1
assert round(comparison["ratio"], 2) == 2.85


# 보고서 본문과 같은 남색 계열. 장식색은 쓰지 않고 한 가지 강조색과 회색만 사용한다.
NAVY = "#123f64"
BLUE = "#397a9f"
MID_BLUE = "#78a9c2"
PALE_BLUE = "#c4d8e5"
MAP_GRAY = "#e7ecef"
TEXT = "#1c2f3d"
MUTED = "#637481"
LINE = "#cbd5db"
GRID = "#e3e9ed"
WHITE = "#ffffff"
BASELINE = "#a0acb5"


# 100m 격자 산출물로 평면 지도를 직접 구성한다.
xy = [tuple(map(int, row["grid_id"].split("_"))) for row in m5]
x0, x1 = min(x for x, _ in xy), max(x for x, _ in xy)
y0, y1 = min(y for _, y in xy), max(y for _, y in xy)
map_image = Image.new("RGB", (x1 - x0 + 1, y1 - y0 + 1), WHITE)
admin_points = defaultdict(list)
map_colors = [MAP_GRAY, PALE_BLUE, MID_BLUE, BLUE, NAVY]

for row, (gx, gy) in zip(m5, xy):
    percentile = float(row["priority_pct"])
    group = 4 if percentile >= 0.95 else 3 if percentile >= 0.90 else 2 if percentile >= 0.80 else 1 if percentile >= 0.70 else 0
    map_image.putpixel((gx - x0, y1 - gy), tuple(bytes.fromhex(map_colors[group][1:])))
    admin_points[(row["sgg_nm"], row["adm_nm"])].append((gx, gy))

map_path = TMP / "최종점검순위_평면지도.png"
map_image.save(map_path)


pdfmetrics.registerFont(TTFont("KR", "C:/Windows/Fonts/malgun.ttf"))
pdfmetrics.registerFont(TTFont("KRB", "C:/Windows/Fonts/malgunbd.ttf"))

W, H = 841.89, 595.28
doc = canvas.Canvas(str(OUT), pagesize=(W, H))
doc.setTitle("수문장 데이터시각화 3D 유지 시안")
doc.setAuthor("수문장")


def text(x, y, value, size=9, color=TEXT, bold=False):
    doc.setFillColor(HexColor(color))
    doc.setFont("KRB" if bold else "KR", size)
    doc.drawString(x, H - y - size * 0.82, value)


def right_text(x, y, value, size=9, color=TEXT, bold=False):
    doc.setFillColor(HexColor(color))
    doc.setFont("KRB" if bold else "KR", size)
    doc.drawRightString(x, H - y - size * 0.82, value)


def center_text(x, y, value, size=9, color=TEXT, bold=False):
    doc.setFillColor(HexColor(color))
    doc.setFont("KRB" if bold else "KR", size)
    doc.drawCentredString(x, H - y - size * 0.82, value)


def rect(x, y, width, height, color, stroke=None, stroke_width=0.7):
    doc.setFillColor(HexColor(color))
    if stroke:
        doc.setStrokeColor(HexColor(stroke))
        doc.setLineWidth(stroke_width)
        doc.rect(x, H - y - height, width, height, fill=1, stroke=1)
    else:
        doc.rect(x, H - y - height, width, height, fill=1, stroke=0)


def rule(x1_, y1_, x2_, y2_, color=LINE, width=0.7):
    doc.setStrokeColor(HexColor(color))
    doc.setLineWidth(width)
    doc.line(x1_, H - y1_, x2_, H - y2_)


def rank_marker(x, y, rank):
    doc.setFillColor(HexColor(NAVY))
    doc.setStrokeColor(HexColor(WHITE))
    doc.setLineWidth(0.8)
    doc.circle(x, H - y, 7.2, fill=1, stroke=1)
    doc.setFillColor(HexColor(WHITE))
    doc.setFont("KRB", 6.6)
    doc.drawCentredString(x, H - y - 2.3, str(rank))


def horizontal_bar_chart(x, y, width, labels_, values, colors, max_value, ticks, bar_gap=29):
    label_w = 82
    plot_x = x + label_w
    plot_w = width - label_w - 36
    chart_top = y + 18
    chart_bottom = chart_top + bar_gap * len(labels_) + 8

    for tick in ticks:
        px = plot_x + plot_w * tick / max_value
        rule(px, chart_top - 10, px, chart_bottom, GRID, 0.45)
        center_text(px, chart_bottom + 4, f"{tick:.0%}", 6.2, MUTED)

    for idx, (label, value, color) in enumerate(zip(labels_, values, colors)):
        by = chart_top + idx * bar_gap
        text(x, by + 2, label, 7.7, TEXT)
        rect(plot_x, by, plot_w * value / max_value, 13, color)
        text(plot_x + plot_w * value / max_value + 6, by + 1, f"{value:.1%}", 8.2, color, True)


# Header: 공모전 보고서 표지와 같은 정보형 제목 체계.
text(34, 21, "2026년 Big Data 활용 대회  |  분야 1. 빅데이터 분석 및 시각화  |  수문장", 7.5, MUTED)
text(34, 43, "부산시 침수위험지역 선제점검 우선순위", 24, NAVY, True)
text(34, 77, "100m 격자 81,224개를 분석하여 호우 전 현장점검이 필요한 지역의 순서를 제안", 9.3, TEXT)
right_text(808, 48, "분석 기준", 7, MUTED)
right_text(808, 61, "침수흔적 2009-2022", 7.5, TEXT, True)
right_text(808, 75, "공간 교차검증(OOF)", 7.5, TEXT, True)
rule(34, 98, 808, 98, NAVY, 1.2)


# Left: 핵심 자산인 3D 격자 시각물을 유지한다.
text(34, 116, "01  3D 격자 기반 선제점검 우선순위", 12.5, NAVY, True)
text(34, 138, "100m 격자의 우선순위와 과거 침수흔적을 입체적으로 확인", 7.8, MUTED)

visual = Image.open(VISUAL_3D)
visual_w = 470
visual_h = visual_w * visual.height / visual.width
doc.drawImage(ImageReader(visual), 31, H - 151 - visual_h, visual_w, visual_h, mask="auto")
visual.close()

text(38, 497, "현장 활용", 7.5, NAVY, True)
text(90, 497, "행정동별 상위 격자 확인  ·  배수로·맨홀 및 저지대 시설 점검  ·  호우 시 인력·장비 우선 배치", 7.5, TEXT)


# Right: 표준 막대그래프와 명시적 분모를 사용한다.
rule(509, 116, 509, 518, LINE, 0.75)
text(531, 116, "02  핵심 검증 결과", 12.5, NAVY, True)

text(531, 145, "같은 772개 지역을 점검했을 때", 9.5, TEXT, True)
text(531, 163, "침수이력 포함 비율은 기존 지정구역 대비 2.85배", 9.8, NAVY, True)
horizontal_bar_chart(
    531,
    181,
    277,
    ["기존 지정구역", "선제점검 모델"],
    [comparison["base"], matched_precision],
    [BASELINE, NAVY],
    0.60,
    [0.0, 0.2, 0.4, 0.6],
)
rule(531, 292, 808, 292, LINE, 0.7)

text(531, 309, "상위 10% 점검 후보의 정확도", 9.3, TEXT, True)
text(531, 333, "21.0%", 22, NAVY, True)
text(614, 339, "1,709 / 8,122개", 7.3, MUTED)
text(531, 368, "모델이 선정한 5곳 중 약 1곳에 과거 침수이력", 8.7, TEXT, True)
text(531, 386, "무작위 점검 기대치 3.6% 대비 약 5.9배", 7.3, MUTED)
rule(531, 398, 808, 398, LINE, 0.7)

text(531, 412, "점검범위별 과거 침수지역 포착률", 9.3, TEXT, True)
horizontal_bar_chart(
    531,
    420,
    277,
    ["상위 10%", "상위 20%"],
    [top10_capture, top20_capture],
    [MID_BLUE, BLUE],
    1.0,
    [0.0, 0.25, 0.5, 0.75, 1.0],
    bar_gap=28,
)
text(531, 515, "상위 20% 점검 시 과거 침수지역 2,893개 중 2,180개 포함", 6.8, MUTED)


# Footer: 해석 범위와 출처를 한 곳에 모은다.
rule(34, 545, 808, 545, LINE, 0.65)
text(34, 555, "주: 성능 수치는 M9 자치구 공간 교차검증(OOF) 결과이며, 지도는 전체학습 감수성에 노출·대응 여건을 결합한 M5 최종순위입니다.", 6.0, MUTED)
text(34, 565, "과거 침수이력에 대한 비교로 미래 피해 저감률을 뜻하지 않으며, 자동 침수경보가 아닌 사전점검 후보 선정에 활용합니다.", 6.0, MUTED)
text(34, 575, "자료: 침수흔적도 2009-2022 · SGIS 인구 2024 · 제출용 활용데이터 04·05 · EPSG:5186", 5.8, MUTED)
right_text(808, 575, "Copernicus WorldDEM-30 © DLR/Airbus · COPERNICUS (EU/ESA)", 5.3, MUTED)

doc.showPage()
doc.save()


pdf = PdfReader(OUT)
assert len(pdf.pages) == 1
extracted = pdf.pages[0].extract_text()
for required in [
    "부산시 침수위험지역 선제점검 우선순위",
    "81,224",
    "772",
    "19.3%",
    "55.1%",
    "2.85배",
    "21.0%",
    "59.1%",
    "75.4%",
    "Copernicus",
]:
    assert required in extracted, required

print(OUT)
