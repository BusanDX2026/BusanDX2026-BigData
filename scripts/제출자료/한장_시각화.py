from pathlib import Path
import csv, json
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'output/활용데이터_CSV/활용데이터_CSV'
OUT=ROOT/'output/pdf/부산_호우_선제점검_데이터시각화_1페이지_수정본.pdf'
OUT.parent.mkdir(parents=True,exist_ok=True)
def read(name):
    with open(DATA/name,encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
d=read('05_M5_최종점검순위_100m격자.csv')
dong=read('06_M5_행정동별_점검순위.csv')
hz=read('04_M9_침수감수성_100m격자.csv')
cfg=json.loads((ROOT/'공공데이터/가공데이터/04_모델/M9_검증.json').read_text(encoding='utf-8'))
n=len(d); k=int(n*.1); positives=sum(int(r['trace_flag']) for r in d)
labels={r['grid_id']:int(r['trace_flag']) for r in d}
top9=sorted(hz,key=lambda r:float(r['hazard_oof']),reverse=True)[:k]
capture=sum(labels[r['grid_id']] for r in top9)/positives
precision=sum(labels[r['grid_id']] for r in top9)/k
top5=sorted(d,key=lambda r:float(r['priority']),reverse=True)[:k]
pop_total=sum(float(r['pop']) for r in d if int(r['trace_flag']))
pop_capture=sum(float(r['pop']) for r in top5 if int(r['trace_flag']))/pop_total
cmp=next(r for r in cfg['hazdist'] if r['flag']=='hazdist_flood')
xy=[tuple(map(int,r['grid_id'].split('_'))) for r in d]
x0=min(x for x,y in xy); x1=max(x for x,y in xy)
y0=min(y for x,y in xy); y1=max(y for x,y in xy)
img=Image.new('RGB',(x1-x0+1,y1-y0+1),'#ffffff')
colors=['#e7ebef','#abc6d9','#498eb0','#123f64']
for r,(x,y) in zip(d,xy):
    p=float(r['priority_pct'])
    g=3 if p>=.95 else 2 if p>=.9 else 1 if p>=.8 else 0
    img.putpixel((x-x0,y1-y),tuple(bytes.fromhex(colors[g][1:])))
pdfmetrics.registerFont(TTFont('KR','C:/Windows/Fonts/malgun.ttf'))
pdfmetrics.registerFont(TTFont('KRB','C:/Windows/Fonts/malgunbd.ttf'))
W,H=841.89,595.28
c=canvas.Canvas(str(OUT),pagesize=(W,H))
c.setTitle('부산 호우 선제점검 우선순위 | 데이터 시각화')
c.setAuthor('부산 호우 선제점검 분석팀')
NAVY='#123f64'; TEXT='#172c3c'; MUTED='#546777'; LINE='#dce4ea'; ORANGE='#cc7940'
def text(x,y,s,size=9,color=TEXT,bold=False):
    c.setFillColor(HexColor(color)); c.setFont('KRB' if bold else 'KR',size); c.drawString(x,H-y-size*.8,s)
def rect(x,y,w,h,color):
    c.setFillColor(HexColor(color)); c.rect(x,H-y-h,w,h,fill=1,stroke=0)
def line(x,y,x2,y2,color=LINE):
    c.setStrokeColor(HexColor(color)); c.setLineWidth(.6); c.line(x,H-y,x2,H-y2)
text(26,20,'2026년 Big Data 활용 대회  |  데이터 시각화',8,MUTED)
text(26,38,'부산 호우 선제점검, 어디부터 시작할까?',23,NAVY,True)
text(26,70,'100m 격자 81,224개에서 물리 감수성을 학습하고, 인구 노출과 대응 여건을 결합한 점검 우선순위',10,MUTED)
line(26,91,816,91,NAVY)
metrics=[(26,'59.1%','침수이력 격자 포착','M9 공간검증 · 상위 10%'),(294,'90.4%','침수이력 격자 거주인구 포함','M5 최종순위 · 상위 10%'),(562,f"{cmp['ratio']:.2f}배",'동일 규모의 침수이력 정밀도','침수 유형 지정구역과 비교')]
for x,value,label,note in metrics:
    text(x,107,value,28,NAVY,True)
    text(x+103,110,label,10,TEXT,True)
    text(x+103,129,note,8,MUTED)
line(26,153,816,153)
text(26,168,'01  부산 전체 점검 우선순위',13,NAVY,True)
text(26,191,'물리 감수성 50% + 노출 35% + 대응결핍 15%',9,MUTED)
# Accurate aspect ratio; grid pixels are 100m squares.
mx,my,mw,mh=27,204,380,302
scale=min(mw/img.width,mh/img.height); iw,ih=img.width*scale,img.height*scale
ix=mx+(mw-iw)/2; iy=my
c.drawImage(ImageReader(img),ix,H-iy-ih,iw,ih,mask='auto')
text(365,214,'N',8,NAVY,True); line(369,230,369,241,NAVY)
bar=50*scale
rect(32,457,bar,2,NAVY); text(32,465,'5 km',7,MUTED)
legend=[('상위 5%',colors[3]),('5~10%',colors[2]),('10~20%',colors[1]),('20% 밖',colors[0])]
for i,(label,col) in enumerate(legend):
    x=29+i*91;rect(x,510,10,7,col);text(x+15,508,label,8,MUTED)
text(26,524,'진한 영역부터 현장점검 후보로 검토합니다.',9,TEXT)
text(26,537,'강서구는 물리 지표의 동일가중 점수를 적용했습니다.',8,MUTED)
line(413,167,413,535)
text(434,168,'02  같은 772개 격자로 비교한 정밀도',13,NAVY,True)
text(434,191,'침수 유형 지정구역 규모에 맞춘 M9 공간검증(OOF)',8.5,MUTED)
for y,label,val,col in [(216,'기존 지정구역',cmp['base'], '#a7b4bf'),(248,'M9 상위 772개',cmp['model'],NAVY)]:
    text(434,y+3,label,9,TEXT)
    rect(534,y,215*val/.6,19,col)
    text(542+215*val/.6,y+3,f'{val:.1%}',10,col,True)
text(434,278,'과거 침수이력 비교이며, 기존 지정구역 보완에 활용합니다.',8,MUTED)
line(434,300,815,300)
text(434,314,'03  우선 점검 행정동 TOP 5',13,NAVY,True)
text(434,337,'순위 기준: 동 내 상위 10% 격자 비율 × 추정 거주인구',8,MUTED)
for i,r in enumerate(dong[:5]):
    y=360+i*26; val=float(r['위험인구'])
    text(434,y,f"{i+1}  {r['sgg_nm']} {r['adm_nm']}",9,TEXT)
    rect(568,y,157*val/28000,12,NAVY if i==0 else '#70a3bf')
    text(733,y,f'{val/10000:.2f}만',9,NAVY,True)
text(434,495,'검증  PR-AUC 0.3059  |  양성률 3.56%  |  상위 10% 정밀도 21.0%',8.4,TEXT,True)
text(434,514,'M9는 공간검증 점수, M5는 전체학습 점수와 노출·대응 여건의 결합입니다.',7.6,MUTED)
text(434,527,'두 결과는 평가 조건이 다르며, 90.4%는 미래 피해 저감률이 아닙니다.',7.6,MUTED)
rect(26,549,790,18,'#edf3f7')
text(37,554,'활용: 후보지 선정 → 배수로·맨홀 현장 확인 → 인력·장비 배치   |   점검 순서 지원용이며 자동 침수경보용이 아닙니다.',8.8,NAVY)
text(26,571,'자료: 제출용 활용데이터 03~07 및 M9 검증 결과 · 침수흔적도 2009~2022 · SGIS 인구 2024 · 좌표계 EPSG:5186',6.5,MUTED)
text(26,580,'produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018',6.2,MUTED)
text(26,588,'provided under COPERNICUS by the European Union and ESA; all rights reserved',6.2,MUTED)
c.showPage();c.save()
assert round(capture*100,1)==59.1 and round(pop_capture*100,1)==90.4
print(OUT)
