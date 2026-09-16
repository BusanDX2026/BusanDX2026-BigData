from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Preformatted
from reportlab.lib.units import mm
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/openlab/02_수문장_코드및실행절차.pdf'
OUT.parent.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('Malgun','C:/Windows/Fonts/malgun.ttf'))
pdfmetrics.registerFont(TTFont('MalgunBD','C:/Windows/Fonts/malgunbd.ttf'))
styles=getSampleStyleSheet(); NAVY=HexColor('#0D5C58'); INK=HexColor('#233936'); MUTED=HexColor('#667874')
styles.add(ParagraphStyle(name='KTitle',fontName='MalgunBD',fontSize=21,leading=28,textColor=NAVY,spaceAfter=10))
styles.add(ParagraphStyle(name='KHead',fontName='MalgunBD',fontSize=14,leading=20,textColor=NAVY,spaceBefore=5,spaceAfter=7))
styles.add(ParagraphStyle(name='KBody',fontName='Malgun',fontSize=9.5,leading=15,textColor=INK,spaceAfter=5))
styles.add(ParagraphStyle(name='KSmall',fontName='Malgun',fontSize=7.6,leading=11,textColor=MUTED))
styles.add(ParagraphStyle(name='KCenter',fontName='MalgunBD',fontSize=11,leading=16,textColor=NAVY,alignment=TA_CENTER))
styles.add(ParagraphStyle(name='CodeBlock',fontName='Courier',fontSize=6.5,leading=8.2,textColor=INK,backColor=HexColor('#F1F5F4'),borderPadding=6))
def P(s,style='KBody'): return Paragraph(s,styles[style])
def hf(c,doc):
    c.saveState(); c.setFont('Malgun',7); c.setFillColor(MUTED); c.drawString(18*mm,10*mm,'수문장 | 오픈랩 반입용 코드·실행절차'); c.drawRightString(192*mm,10*mm,f'{doc.page} / 4'); c.restoreState()
doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=17*mm,bottomMargin=17*mm)
story=[P('수문장 오픈랩 코드·실행절차','KTitle'),P('부산 호우 선제점검 우선순위 분석','KHead'),P('이 문서는 오픈랩에서 올인원 Excel을 읽어 동일한 M9 침수 감수성 모델과 M5 최종 점검순위를 재현하기 위한 실행 안내서입니다. Excel 파일 자체는 매크로가 아니며, 오픈랩의 Python 환경에서 아래 코드를 실행합니다.','KBody'),Spacer(1,5)]
data=[[P('반입 파일'),P('역할')],[P('01_수문장_올인원_분석데이터.xlsx'),P('모델입력·기존결과·변수정의·데이터출처·행정동 결과를 포함한 통합 입력자료')],[P('02_수문장_코드및실행절차.pdf'),P('아래 실행 순서와 핵심 코드')],[P('03_수문장_데이터시각화.pdf'),P('지도 및 핵심 결과 설명')]]
t=Table(data,colWidths=[55*mm,115*mm]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),'white'),('GRID',(0,0),(-1,-1),.3,HexColor('#C9D8D4')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6)])); story += [t,Spacer(1,12)]
story += [P('분석 구조','KHead'),P('① 모델입력 시트에서 100m 격자 81,224개와 정규화된 16개 물리 피처를 읽습니다. ② 자치구 GroupKFold 5분할로 미학습 격자 OOF 점수를 산출하고 PR-AUC를 계산합니다. ③ 전체 자료로 다시 학습한 감수성 백분위에 노출 35%, 대응결핍 15%를 결합해 M5 순위를 만듭니다. ④ 강서구는 기록 편향 보정으로 물리 MCDA 감수성을 적용합니다.','KBody'),P('현재 산출물 기준값: PR-AUC 0.3059, 양성률 3.56%, 상위 10% 침수이력 포착 59.1%, 동일한 772개 침수 유형 지정구역 비교 19.3% → 55.1%.','KBody'),PageBreak()]
story += [P('1. 오픈랩 실행 준비','KTitle'),P('필요 환경','KHead'),P('Python 3.10 이상과 pandas, numpy, scikit-learn, scipy, xgboost, openpyxl이 필요합니다.','KBody'),P('pip install pandas numpy scikit-learn scipy xgboost openpyxl','CodeBlock'),P('Excel 파일과 실행 코드를 같은 폴더에 두고, 코드의 EXCEL_PATH를 실제 파일명으로 맞춥니다. 시트 이름은 모델입력이며 열 이름은 변경하지 않습니다.','KBody'),P('실행 순서','KHead'),P('1) Excel 업로드 → 2) 패키지 확인 → 3) M9 코드 실행 → 4) M5 코드 실행 → 5) 생성된 CSV와 Excel의 기존결과 시트 비교','KBody'),P('M9는 grid_id, sgg_cd, trace_flag와 16개 _s 열을 사용합니다. M5는 노출 10개 _s 열, 대응결핍 2개 _s 열, sgg_nm, pop, hazdist_flood_active를 사용합니다.','KBody'),P('hazard_oof는 공간 교차검증 성능평가용입니다. hazard_pct와 priority는 전체 자료로 학습한 결과를 정책용 순위에 사용합니다. 인구·건물·펌프장 값은 M9 학습 피처가 아닙니다.','KBody'),PageBreak()]
m9='''import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

EXCEL_PATH = "01_수문장_올인원_분석데이터.xlsx"
df = pd.read_excel(EXCEL_PATH, sheet_name="모델입력")
FEATS = [
 "elev_min_s", "slope_mean_s", "tpi_s", "lowland3_ratio_s",
 "fluv_area_ratio_s", "rain_annmax_mm_s", "flow_acc_log_s",
 "twi_s", "dist_stream_m_s", "imperv_ratio_s", "agri_ratio_s",
 "paddy_ratio_s", "forest_ratio_s", "water_ratio_s", "road_ratio_s",
 "mh_no_dredge_ratio_s"
]
X, y, groups = df[FEATS], df.trace_flag.astype(int), df.sgg_cd
params = dict(
 max_depth=3, n_estimators=600, learning_rate=0.03,
 min_child_weight=30, subsample=0.7, colsample_bytree=0.6,
 scale_pos_weight=27.076045627376427, eval_metric="aucpr",
 tree_method="hist", reg_lambda=1.0, n_jobs=-1, random_state=42
)
oof = np.zeros(len(df))
for train, test in GroupKFold(n_splits=5).split(X, y, groups):
 m = XGBClassifier(**params)
 m.fit(X.iloc[train], y.iloc[train])
 oof[test] = m.predict_proba(X.iloc[test])[:, 1]
print("PR-AUC", average_precision_score(y, oof), "base rate", y.mean())
final = XGBClassifier(**params); final.fit(X, y)
out = df[["grid_id", "sgg_nm", "adm_nm", "trace_flag", "pop",
          "hazdist_flood_active"]].copy()
out["hazard_raw"] = final.predict_proba(X)[:, 1]
out["hazard_oof"] = oof
out["hazard_pct"] = out.hazard_raw.rank(pct=True)
out.to_csv("M9_재실행결과.csv", index=False, encoding="utf-8-sig")'''
story += [P('2. M9 침수 감수성 실행문','KTitle'),P('정규화된 물리 피처 16개만 사용합니다. 실행 후 PR-AUC가 0.3059 부근인지 확인합니다. 패키지 버전에 따라 소폭 달라질 수 있습니다.','KBody'),Preformatted(m9,styles['CodeBlock']),PageBreak()]
m5='''import numpy as np, pandas as pd
m = pd.read_excel("01_수문장_올인원_분석데이터.xlsx", sheet_name="모델입력")
r = pd.read_csv("M9_재실행결과.csv")
EXPO = ["pop_s", "pop_65_s", "old_ratio_s", "bldg_cnt_s",
 "floor_area_s", "resid_floor_area_s", "basement_bldg_s",
 "max_floors_s", "underpass_n_300m_s", "underpass_len_m_s"]
CAP = ["pump_dist_m_s", "pump_n_1000m_s"]
HAZ = ["elev_min_s", "slope_mean_s", "tpi_s", "lowland3_ratio_s",
 "fluv_area_ratio_s", "rain_annmax_mm_s", "flow_acc_log_s", "twi_s",
 "dist_stream_m_s", "imperv_ratio_s", "agri_ratio_s", "paddy_ratio_s",
 "forest_ratio_s", "water_ratio_s", "road_ratio_s", "mh_no_dredge_ratio_s"]
r["exposure_pct"] = m[EXPO].mean(axis=1).rank(pct=True)
r["capacity_pct"] = m[CAP].mean(axis=1).rank(pct=True)
r["hazard_mcda_pct"] = m[HAZ].mean(axis=1).rank(pct=True)
r["priority_ml"] = .50*r.hazard_pct + .35*r.exposure_pct + .15*r.capacity_pct
r["priority_mcda"] = .50*r.hazard_mcda_pct + .35*r.exposure_pct + .15*r.capacity_pct
r["priority_src"] = np.where(r.sgg_nm == "강서구", "MCDA(물리)", "ML")
r["priority"] = np.where(r.sgg_nm == "강서구", r.priority_mcda, r.priority_ml)
r["priority_pct"] = r.priority.rank(pct=True)
r.to_csv("M5_재실행결과.csv", index=False, encoding="utf-8-sig")
print(r.sort_values("priority", ascending=False).head(20))'''
story += [P('3. M5 최종 점검순위 실행문','KTitle'),P('M9 결과는 Excel과 같은 행 순서를 유지해야 합니다. 실제 운영에서는 grid_id로 병합하고 순위 결과를 기존결과 시트와 비교합니다.','KBody'),Preformatted(m5,styles['CodeBlock']),Spacer(1,5),P('재실행 후 M9_재실행결과.csv와 M5_재실행결과.csv의 행 수, grid_id 중복, 상위 행정동을 확인합니다.','KSmall')]
doc.build(story,onFirstPage=hf,onLaterPages=hf)
assert len(PdfReader(OUT).pages)==4
print(OUT)
