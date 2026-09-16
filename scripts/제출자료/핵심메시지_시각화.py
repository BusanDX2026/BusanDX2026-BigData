from pathlib import Path
import csv
import json
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
OUT = ROOT / "output/pdf/수문장_데이터시각화_핵심메시지_1장.pdf"
TMP = ROOT / "tmp/pdfs/핵심메시지"
TMP.mkdir(parents=True, exist_ok=True)
OUT.parent.mkdir(parents=True, exist_ok=True)


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
top10_k = int(len(m9) * 0.10)
top20_k = int(len(m9) * 0.20)
positives = sum(labels.values())
oof_order = sorted(m9, key=lambda row: float(row["hazard_oof"]), reverse=True)
top10_hits = sum(labels[row["grid_id"]] for row in oof_order[:top10_k])
top20_hits = sum(labels[row["grid_id"]] for row in oof_order[:top20_k])
capture = top10_hits / positives
top20_capture = top20_hits / positives
matched_hits = sum(labels[row["grid_id"]] for row in oof_order[: comparison["k"]])
matched_precision = matched_hits / comparison["k"]

assert len(m5) == 81224
assert positives == 2893
assert top10_k == 8122 and top10_hits == 1709
assert top20_k == 16244 and top20_hits == 2180
assert round(capture * 100, 1) == 59.1
assert round(top20_capture * 100, 1) == 75.4
assert comparison["k"] == 772
assert abs(matched_precision - comparison["model"]) < 1e-12
assert round(comparison["base"] * 100, 1) == 19.3
assert round(matched_precision * 100, 1) == 55.1
assert round(comparison["ratio"], 2) == 2.85


# 보고서에 사용한 남색 계열을 유지한다. 강조색을 늘리지 않고 명도 차이로 순위를 표현한다.
NAVY = "#123f64"
BLUE = "#397a9f"
MID_BLUE = "#78a9c2"
PALE_BLUE = "#c4d8e5"
MAP_GRAY = "#e7ecef"
TEXT = "#1c2f3d"
MUTED = "#607282"
LINE = "#d6e0e6"
LIGHT = "#edf3f6"
WHITE = "#ffffff"
BASELINE = "#9eabb5"


# 100m 격자 자체를 픽셀로 사용하여 행정경계를 임의로 그리지 않는다.
xy = [tuple(map(int, row["grid_id"].split("_"))) for row in m5]
x0, x1 = min(x for x, _ in xy), max(x for x, _ in xy)
y0, y1 = min(y for _, y in xy), max(y for _, y in xy)
map_image = Image.new("RGB", (x1 - x0 + 1, y1 - y0 + 1), WHITE)
district_points = defaultdict(list)
admin_points = defaultdict(list)
map_colors = [MAP_GRAY, PALE_BLUE, MID_BLUE, BLUE, NAVY]

for row, (gx, gy) in zip(m5, xy):
    percentile = float(row["priority_pct"])
    group = 4 if percentile >= 0.95 else 3 if percentile >= 0.90 else 2 if percentile >= 0.80 else 1 if percentile >= 0.70 else 0
    map_image.putpixel((gx - x0, y1 - gy), tuple(bytes.fromhex(map_colors[group][1:])))
    district_points[row["sgg_nm"]].append((gx, gy))
    admin_points[(row["sgg_nm"], row["adm_nm"])].append((gx, gy))

map_path = TMP / "최종점검순위_지도.png"
map_image.save(map_path)


pdfmetrics.registerFont(TTFont("KR", "C:/Windows/Fonts/malgun.ttf"))
pdfmetrics.registerFont(TTFont("KRB", "C:/Windows/Fonts/malgunbd.ttf"))

W, H = 841.89, 595.28  # A4 landscape
doc = canvas.Canvas(str(OUT), pagesize=(W, H))
doc.setTitle("수문장 데이터 시각화 - 부산 호우 선제점검 우선순위")
doc.setAuthor("수문장")


def text(x, y, value, size=9, color=TEXT, bold=False):
    doc.setFillColor(HexColor(color))
    doc.setFont("KRB" if bold else "KR", size)
    doc.drawString(x, H - y - size * 0.82, value)


def right_text(x, y, value, size=9, color=TEXT, bold=False):
    doc.setFillColor(HexColor(color))
    doc.setFont("KRB" if bold else "KR", size)
    doc.drawRightString(x, H - y - size * 0.82, value)


def rect(x, y, width, height, color):
    doc.setFillColor(HexColor(color))
    doc.rect(x, H - y - height, width, height, fill=1, stroke=0)


def rule(x1_, y1_, x2_, y2_, color=LINE, width=0.7):
    doc.setStrokeColor(HexColor(color))
    doc.setLineWidth(width)
    doc.line(x1_, H - y1_, x2_, H - y2_)


def rank_marker(x, y, rank):
    doc.setFillColor(HexColor(NAVY))
    doc.circle(x, H - y, 8, fill=1, stroke=0)
    doc.setFillColor(HexColor(WHITE))
    doc.setFont("KRB", 7.3)
    doc.drawCentredString(x, H - y - 2.6, str(rank))


# Header: 제목만 읽어도 분석 대상과 목적이 드러나도록 쓴다.
text(32, 20, "2026년 Big Data 활용 대회  |  수문장", 8.2, MUTED)
text(32, 39, "부산시 침수위험지역", 25, TEXT, True)
text(32, 70, "선제점검 우선순위", 28, NAVY, True)
text(32, 106, "모든 지역을 동시에 점검하기 어려워, 호우 전에 먼저 확인할 100m 격자 순위를 제안합니다.", 10.2, MUTED)
rule(32, 126, 810, 126, NAVY, 1.2)


# Left: 운영 결과 지도. 지도가 본문 면적의 절반 이상을 차지하도록 한다.
text(32, 143, "호우 전, 어디부터 점검할까?", 14, NAVY, True)
text(32, 166, "물리적 감수성에 인구·건물 노출과 대응 여건을 더한 100m 격자 순위", 8.8, MUTED)

map_x, map_y, map_w, map_h = 40, 187, 410, 278
scale = min(map_w / map_image.width, map_h / map_image.height)
draw_w, draw_h = map_image.width * scale, map_image.height * scale
draw_x = map_x + (map_w - draw_w) / 2
draw_y = map_y + (map_h - draw_h) / 2
doc.drawImage(ImageReader(map_path), draw_x, H - draw_y - draw_h, draw_w, draw_h, mask="auto")

# 행정동 상위 5위를 지도 안에서 번호와 이름으로 함께 확인할 수 있게 한다.
text(45, 205, "행정동 점검순위", 8.2, NAVY, True)
for idx, row in enumerate(dong_ranking[:5], start=1):
    text(45, 223 + (idx - 1) * 18, f"{idx}  {row['sgg_nm']} {row['adm_nm']}", 7.5, TEXT, idx == 1)

marker_offsets = {1: (9, 1), 2: (-10, 9), 3: (-12, -4), 4: (0, 0), 5: (8, -7)}
for idx, row in enumerate(dong_ranking[:5], start=1):
    points = admin_points[(row["sgg_nm"], row["adm_nm"])]
    gx = sum(x for x, _ in points) / len(points)
    gy = sum(y for _, y in points) / len(points)
    px = draw_x + (gx - x0 + 0.5) * scale
    py = draw_y + (y1 - gy + 0.5) * scale
    dx, dy = marker_offsets[idx]
    rank_marker(px + dx, py + dy, idx)

text(426, 195, "N", 8, NAVY, True)
rule(430, 210, 430, 231, NAVY, 1)
rule(430, 210, 427, 216, NAVY, 1)
rule(430, 210, 433, 216, NAVY, 1)

legend = [("상위 5%", NAVY), ("5~10%", BLUE), ("10~20%", MID_BLUE), ("20~30%", PALE_BLUE), ("30% 밖", MAP_GRAY)]
for idx, (label, color) in enumerate(legend):
    lx = 36 + idx * 83
    rect(lx, 472, 10, 8, color)
    text(lx + 14, 470, label, 7.8, MUTED)
text(35, 491, "진한 영역부터 배수로·맨홀과 저지대 시설을 현장에서 확인합니다.", 8.8, TEXT)
text(35, 506, "행정동 순위 기준: 동 내 상위 10% 격자 비율 × 추정 거주인구", 7.6, MUTED)


# Right: 전문지표보다 일반인이 바로 이해할 수 있는 비율 문장을 먼저 보여준다.
rule(476, 143, 476, 515, LINE, 0.8)
text(501, 143, "분석 결과", 14, NAVY, True)

# 1. 기존 기준과의 동일 면적 비교를 가장 먼저 제시한다.
text(501, 174, "기존 침수 유형 지정구역 대비", 10.2, TEXT, True)
text(501, 196, "예측모델의 침수지역 포함 비율 약 2.85배", 12.2, NAVY, True)
text(501, 221, "같은 수인 772개 격자를 기준으로 비교", 7.8, MUTED)

bar_x, bar_w = 625, 132
for y, label, value, color in [
    (244, "기존 지정구역", comparison["base"], BASELINE),
    (274, "감수성 모델", matched_precision, NAVY),
]:
    text(501, y + 2, label, 8.3, TEXT)
    rect(bar_x, y, bar_w, 15, LIGHT)
    rect(bar_x, y, bar_w * value / 0.60, 15, color)
    right_text(807, y + 1, f"{value:.1%}", 9.5, color, True)

rule(501, 306, 808, 306, LINE, 0.7)

# 2. 상위 10% 후보의 정밀도를 일상적인 빈도로 번역한다.
text(501, 321, "과거 침수이력 기준 공간검증 결과,", 8.8, MUTED)
text(501, 341, "모델이 예측한 5곳 중 약 1곳이", 12.4, NAVY, True)
text(501, 363, "과거 침수지역이었습니다", 10.2, TEXT, True)
for idx in range(5):
    rect(502 + idx * 37, 388, 28, 19, NAVY if idx == 0 else LIGHT)
text(701, 390, "21.0%", 10.5, NAVY, True)
text(501, 416, "감수성 상위 10%인 8,122개 중 1,709개", 7.7, MUTED)

rule(501, 440, 808, 440, LINE, 0.7)

# 3. 점검 자원을 20%까지 투입할 때의 운영 시나리오를 별도로 표시한다.
text(501, 454, "예측모델의 상위 20% 지역이", 9.8, TEXT, True)
text(501, 475, "과거 침수지역의 75.4% 포함", 13.2, BLUE, True)
rect(502, 503, 252, 12, LIGHT)
rect(502, 503, 252 * top20_capture, 12, BLUE)
right_text(807, 501, "75.4%", 10.2, BLUE, True)
text(501, 519, "16,244개 점검 시 2,893개 중 2,180개", 7.2, MUTED)


# Bottom: 분석 결과가 실제 업무로 이어지는 한 줄 흐름.
rect(32, 530, 778, 22, LIGHT)
text(47, 536, "호우 예보", 8.8, NAVY, True)
text(135, 536, "→", 8.8, MUTED)
text(166, 536, "행정동별 상위 격자 확인", 8.8, NAVY, True)
text(332, 536, "→", 8.8, MUTED)
text(363, 536, "배수로·맨홀 현장점검", 8.8, NAVY, True)
text(520, 536, "→", 8.8, MUTED)
text(551, 536, "인력·장비 우선 배치", 8.8, NAVY, True)

text(32, 559, "평가 수치는 M9 자치구 공간 교차검증(OOF), 지도는 전체학습 감수성에 노출·대응 여건을 결합한 M5 최종순위입니다.", 6.4, MUTED)
text(32, 568, "과거 침수이력에 대한 비교이며 미래 피해 저감률이 아닙니다. 자동 침수경보가 아닌 사전점검 후보 선정용입니다.", 6.4, MUTED)
text(32, 577, "자료: 침수흔적도 2009~2022 · SGIS 인구 2024 · 제출용 활용데이터 04·05 · EPSG:5186", 6.2, MUTED)
text(516, 577, "Copernicus WorldDEM-30 © DLR/Airbus, provided under COPERNICUS by EU and ESA", 5.7, MUTED)

doc.showPage()
doc.save()


pdf = PdfReader(OUT)
assert len(pdf.pages) == 1
extracted = pdf.pages[0].extract_text()
for required in ["부산시 침수위험지역", "8,122", "1,709", "21.0%", "16,244", "2,180", "75.4%", "772", "19.3%", "55.1%", "2.85배", "Copernicus"]:
    assert required in extracted, required

print(OUT)
print(
    json.dumps(
        {
            "pages": 1,
            "grid_count": len(m5),
            "top10_k": top10_k,
            "top10_hits": top10_hits,
            "capture": capture,
            "top20_k": top20_k,
            "top20_hits": top20_hits,
            "top20_capture": top20_capture,
            "comparison_k": comparison["k"],
            "comparison_base": comparison["base"],
            "comparison_model": matched_precision,
        },
        ensure_ascii=False,
    )
)
