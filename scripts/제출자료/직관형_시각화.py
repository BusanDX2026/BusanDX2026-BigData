from pathlib import Path
import csv, json
from collections import defaultdict
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'output/활용데이터_CSV/활용데이터_CSV'
OUT=ROOT/'output/pdf/수문장_데이터시각화_직관형_1장.pdf'
TMP=ROOT/'tmp/pdfs/직관형'; TMP.mkdir(parents=True,exist_ok=True)
def read(name):
    with (DATA/name).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
d=read('05_M5_최종점검순위_100m격자.csv')
hz=read('04_M9_침수감수성_100m격자.csv')
cfg=json.loads((ROOT/'공공데이터/가공데이터/04_모델/M9_검증.json').read_text(encoding='utf-8'))
cmp=next(r for r in cfg['hazdist'] if r['flag']=='hazdist_flood')
labels={r['grid_id']:int(r['trace_flag']) for r in d}
k=int(len(d)*.1); npos=sum(labels.values())
order=sorted(hz,key=lambda r:float(r['hazard_oof']),reverse=True)
capture=sum(labels[r['grid_id']] for r in order[:k])/npos
match=sum(labels[r['grid_id']] for r in order[:cmp['k']])/cmp['k']
assert len(d)==81224 and npos==2893 and round(capture*100,1)==59.1
assert cmp['k']==772 and abs(match-cmp['model'])<1e-12
assert round(cmp['base']*100,1)==19.3 and round(match*100,1)==55.1
COL=['#eef2f1','#cfe3e0','#7bbdb5','#2b8c83','#0d5c58']
TEAL=COL[-1]; INK='#203a36'; MUTED='#61736f'; LIGHT='#f3f7f6'; GRAY='#71839b'
xy=[tuple(map(int,r['grid_id'].split('_'))) for r in d]
x0=min(x for x,y in xy); x1=max(x for x,y in xy)
y0=min(y for x,y in xy); y1=max(y for x,y in xy)
img=Image.new('RGB',(x1-x0+1,y1-y0+1),'white')
groups=defaultdict(list)
for r,(x,y) in zip(d,xy):
    p=1-float(r['priority_pct'])+1/len(d)
    g=4 if p<=.05 else 3 if p<=.10 else 2 if p<=.20 else 1 if p<=.30 else 0
    img.putpixel((x-x0,y1-y),tuple(bytes.fromhex(COL[g][1:])))
    groups[r['sgg_nm']].append((x,y))
img.save(TMP/'점검순위_지도.png')
pdfmetrics.registerFont(TTFont('KR','C:/Windows/Fonts/malgun.ttf'))
pdfmetrics.registerFont(TTFont('KRB','C:/Windows/Fonts/malgunbd.ttf'))
W,H=841.89,595.28
c=canvas.Canvas(str(OUT),pagesize=(W,H))
c.setTitle('수문장 | 비가 오기 전, 부산 어디부터 점검할까?')
c.setAuthor('수문장')
def txt(x,y,s,size=10,col=INK,bold=False):
    c.setFont('KRB' if bold else 'KR',size); c.setFillColor(HexColor(col)); c.drawString(x,H-y-size*.82,s)
def box(x,y,w,h,col,r=0):
    c.setFillColor(HexColor(col)); c.roundRect(x,H-y-h,w,h,r,fill=1,stroke=0)
def line(x,y,x2,y2,col='#d9e4e0',width=.7):
    c.setStrokeColor(HexColor(col)); c.setLineWidth(width); c.line(x,H-y,x2,H-y2)
txt(27,19,'2026 Big Data 활용 대회  /  수문장',8.5,MUTED,True)
txt(27,39,'비가 오기 전, 부산 어디부터 점검할까?',25,TEAL,True)
txt(28,76,'침수 가능성에 사람·건물과 대응 여건을 더해, 먼저 살펴볼 곳을 제안합니다.',11,MUTED)
line(27,99,815,99,TEAL,1.1)

txt(28,115,'01  먼저 살펴볼 곳',14,TEAL,True)
txt(28,139,'최종 점검순위 · 100m × 100m 격자 81,224개',9.3,MUTED)
mx,my,mw,mh=28,159,449,307
scale=min(mw/img.width,mh/img.height); iw,ih=img.width*scale,img.height*scale
ix=mx+(mw-iw)/2; iy=my+(mh-ih)/2
c.drawImage(ImageReader(img),ix,H-iy-ih,iw,ih,mask='auto')
# Labels use the geographic centre of each district's included grid cells.
for name in ['강서구','기장군','금정구','해운대구','사하구','부산진구']:
    pts=groups[name]; gx=sum(x for x,y in pts)/len(pts); gy=sum(y for x,y in pts)/len(pts)
    px=ix+(gx-x0+.5)*scale; py=iy+(y1-gy+.5)*scale
    tw=pdfmetrics.stringWidth(name,'KRB',9)
    box(px-tw/2-4,py-5,tw+8,15,'#ffffff',3); txt(px-tw/2,py-2,name,9,INK,True)
txt(445,167,'N',9,TEAL,True); line(449,184,449,207,TEAL,1)
line(449,184,446,190,TEAL,1); line(449,184,452,190,TEAL,1)
line(42,442,42+50*scale,442,TEAL,2); txt(42,449,'5 km',7.5,MUTED)
txt(31,465,'진한 청록색일수록 먼저 점검할 후보입니다.',10,TEAL,True)
for i,(label,color) in enumerate(zip(['상위 5%','5~10%','10~20%','20~30%','30% 밖'],reversed(COL))):
    x=31+i*87; box(x,487,10,8,color); txt(x+14,485,label,8.2,MUTED)
line(491,115,491,498)

txt(513,115,'02  과거 기록으로 확인한 선별 성능',13,TEAL,True)
txt(513,142,'전체 격자 중 상위 10%를 고르면',11,INK,True)
for i in range(10): box(514+i*28,163,23,18,TEAL if i==0 else COL[0],2)
txt(514,190,'과거 침수 기록이 있는 격자의',10,MUTED)
txt(512,209,'59.1%',39,TEAL,True)
txt(658,230,'가 포함됐습니다',11,INK,True)
box(514,260,284,10,COL[0],3); box(514,260,284*capture,10,COL[3],3)
txt(514,280,'포함 59.1%',9,TEAL,True); txt(679,280,'미포함 40.9%',9,MUTED)
txt(514,299,'감수성 모델의 검증 결과이며, 왼쪽 지도 순위와 구분됩니다.',7.5,MUTED)
line(513,320,812,320)

txt(513,334,'03  같은 수의 격자를 비교하면?',13,TEAL,True)
txt(513,358,'각각 772개 중, 과거 침수 기록이 있는 비율',9.5,MUTED)
for yy,label,value,color in [(385,'기존 침수 유형 지정구역',cmp['base'],GRAY),(432,'모델이 선정한 상위 격자',match,TEAL)]:
    txt(514,yy,label,9,INK)
    box(514,yy+17,235,15,COL[0],2)
    box(514,yy+17,235*value,15,color,2)
    txt(758,yy+18,f'{value:.1%}',10,color,True)
txt(514,482,f'침수이력 비율 약 {cmp["ratio"]:.2f}배',12,TEAL,True)

box(27,511,788,31,TEAL,5)
txt(40,521,'기존 점검대상과 함께 검토',10.5,'#ffffff',True)
txt(267,521,'→  배수시설 현장 확인',10.5,'#ffffff',True)
txt(510,521,'→  인력·장비 배치 우선순위 결정',10.5,'#ffffff',True)
txt(28,549,'평가: 수치는 감수성 모델(M9)의 자치구 공간 교차검증(OOF) 결과입니다. 지도는 전체학습 감수성에 노출·대응 여건을 결합한 최종순위(M5)입니다.',6.6,MUTED)
txt(28,558,'유의: 과거 침수이력에 대한 평가이며 미래 피해 저감률이 아닙니다. 자동 침수경보용이 아니며, 강서구 감수성은 물리지표 동일가중 점수를 사용합니다.',6.6,MUTED)
txt(28,567,'자료: 제출용 활용데이터 04·05 및 M9 검증 결과 / 침수흔적도 2009~2022 / SGIS 인구 2024 / 좌표계 EPSG:5186',6.6,MUTED)
txt(28,576,'produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018',6.2,MUTED)
txt(28,584,'provided under COPERNICUS by the European Union and ESA; all rights reserved',6.2,MUTED)
c.showPage(); c.save()
pdf=PdfReader(OUT); assert len(pdf.pages)==1
text=pdf.pages[0].extract_text()
for term in ['59.1%','19.3%','55.1%','772','2.85','Copernicus']: assert term in text,term
print(OUT)
print(json.dumps({'pages':1,'grids':len(d),'positive_grids':npos,'top10_k':k,'capture':capture,'comparison_k':cmp['k'],'comparison_model':match},ensure_ascii=False))
