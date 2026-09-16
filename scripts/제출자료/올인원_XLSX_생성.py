from pathlib import Path
import json
import pandas as pd
import xlsxwriter

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "output" / "활용데이터_CSV" / "활용데이터_CSV"
OUTDIR = ROOT / "output" / "openlab"
OUT = OUTDIR / "01_수문장_올인원_분석데이터.xlsx"
OUTDIR.mkdir(parents=True, exist_ok=True)

input_df = pd.read_csv(DATA / "03_모델입력_100m격자.csv", encoding="utf-8-sig", low_memory=False)
m9 = pd.read_csv(DATA / "04_M9_침수감수성_100m격자.csv", encoding="utf-8-sig", low_memory=False)
m5 = pd.read_csv(DATA / "05_M5_최종점검순위_100m격자.csv", encoding="utf-8-sig", low_memory=False)
sources = pd.read_csv(DATA / "01_원천데이터_출처.csv", encoding="utf-8-sig")
defs = pd.read_csv(DATA / "02_변수정의.csv", encoding="utf-8-sig", low_memory=False)
dong = pd.read_csv(DATA / "06_M5_행정동별_점검순위.csv", encoding="utf-8-sig", low_memory=False)
fire = pd.read_csv(DATA / "07_119_보조검증_행정동.csv", encoding="utf-8-sig", low_memory=False)
compare = pd.read_csv(DATA / "08_지정구역_비교후보.csv", encoding="utf-8-sig", low_memory=False)
assert len(input_df) == len(m9) == len(m5) == 81224
assert input_df.grid_id.is_unique and (input_df.grid_id.values == m9.grid_id.values).all() and (input_df.grid_id.values == m5.grid_id.values).all()
assert int(input_df.trace_flag.sum()) == 2893

feats = ["elev_min_s","slope_mean_s","tpi_s","lowland3_ratio_s","fluv_area_ratio_s","rain_annmax_mm_s","flow_acc_log_s","twi_s","dist_stream_m_s","imperv_ratio_s","agri_ratio_s","paddy_ratio_s","forest_ratio_s","water_ratio_s","road_ratio_s","mh_no_dredge_ratio_s"]
expo = ["pop_s","pop_65_s","old_ratio_s","bldg_cnt_s","floor_area_s","resid_floor_area_s","basement_bldg_s","max_floors_s","underpass_n_300m_s","underpass_len_m_s"]
cap = ["pump_dist_m_s","pump_n_1000m_s"]
input_cols = ["grid_id","sgg_cd","sgg_nm","adm_cd","adm_nm","x_cen","y_cen","trace_flag","trace_count","trace_last_year","trace_max_depth","hazdist_flag","hazdist_flood_active",*feats,*expo,*cap,"pop"]
input_df = input_df[input_cols]
result_cols = ["grid_id","sgg_nm","adm_nm","hazard_raw","hazard_oof","hazard_pct","hazard_mcda_pct","exposure_pct","capacity_pct","priority","priority_pct","priority_src","pop","trace_flag","hazdist_flood_active"]
results = m5.merge(m9[["grid_id","hazard_raw","hazard_oof"]], on="grid_id", how="left", validate="one_to_one")
results = results[result_cols]

def write_df(wb, name, df, widths=None, freeze=True):
    ws = wb.add_worksheet(name)
    ws.hide_gridlines(2)
    header = wb.add_format({"bold": True, "font_name": "Arial", "font_size": 10, "font_color": "white", "bg_color": "#0D5C58", "align": "center", "valign": "vcenter", "text_wrap": True, "border": 0})
    body = wb.add_format({"font_name": "Arial", "font_size": 9, "font_color": "#233936", "valign": "vcenter"})
    num = wb.add_format({"font_name": "Arial", "font_size": 9, "font_color": "#233936", "num_format": "0.0000"})
    for j, col in enumerate(df.columns): ws.write(0, j, col, header)
    ws.set_row(0, 30)
    for start in range(0, len(df), 5000):
        chunk = df.iloc[start:start+5000]
        for i, row in enumerate(chunk.itertuples(index=False, name=None), start+1):
            for j, val in enumerate(row):
                if pd.isna(val): ws.write_blank(i, j, None, body)
                elif isinstance(val, (int, float)) and not isinstance(val, bool): ws.write_number(i, j, float(val), num if ("pct" in str(df.columns[j]) or str(df.columns[j]).endswith("_s") or str(df.columns[j]) in {"priority","hazard_raw","hazard_oof","hazard_mcda_pct","exposure_pct","capacity_pct","old_ratio","trace_max_depth"}) else body)
                else: ws.write(i, j, val, body)
    if freeze: ws.freeze_panes(1, 1)
    for j, col in enumerate(df.columns):
        width = (widths or {}).get(j, 14)
        if col in {"grid_id","sgg_nm","adm_nm"}: width = max(width, 16)
        ws.set_column(j, j, width)
    return ws

wb = xlsxwriter.Workbook(str(OUT), {"constant_memory": True, "strings_to_urls": False})
wb.set_properties({"title":"수문장 오픈랩 올인원 분석데이터", "author":"수문장", "comments":"실제 분석 입력·산출·정의·출처 묶음"})
ws = wb.add_worksheet("요약"); ws.hide_gridlines(2); ws.set_column("A:A", 29); ws.set_column("B:B", 22); ws.set_column("C:H", 15)
title=wb.add_format({"font_name":"Arial","font_size":16,"bold":True,"font_color":"#0D5C58"}); band=wb.add_format({"font_name":"Arial","font_size":11,"bold":True,"font_color":"white","bg_color":"#0D5C58"}); label=wb.add_format({"font_name":"Arial","font_size":10,"bold":True,"font_color":"#233936","bg_color":"#E8F2F0"}); body=wb.add_format({"font_name":"Arial","font_size":10,"font_color":"#233936","text_wrap":True,"valign":"vcenter"}); pct=wb.add_format({"font_name":"Arial","font_size":10,"num_format":"0.0%"})
ws.merge_range("A2:H2","수문장 오픈랩 분석데이터",title); ws.set_row(1,30)
ws.write_row(3,0,["항목","값"],band)
summary_rows=[["100m 격자", "=COUNTA('모델입력'!A2:A81225)", 81224],["침수이력 격자", "=SUM('모델입력'!H2:H81225)", 2893],["M9 학습 피처", 16, None],["M5 결합 지표", 12, None],["좌표계", "EPSG:5186", None]]
for rr,(lab,val,cache) in enumerate(summary_rows,4):
    ws.write(rr,0,lab,label)
    if isinstance(val,str) and val.startswith('='): ws.write_formula(rr,1,val,body,cache)
    else: ws.write(rr,1,val,body)
ws.merge_range("A11:H11","사용 순서",band); steps=["1. 코드·실행절차 PDF와 이 통합문서를 같은 폴더에 둡니다.","2. Python에서 '모델입력' 시트를 읽어 M9 공간 교차검증과 전체학습을 실행합니다.","3. 감수성 백분위와 노출·대응결핍 백분위를 결합해 M5 최종순위를 만듭니다.","4. 오픈랩 데이터는 grid_id, EPSG:5186 좌표 또는 행정동 코드로 결합합니다.","5. '기존결과' 시트와 재실행 결과를 대조합니다."]; 
for i,s in enumerate(steps,11): ws.merge_range(i,0,i,7,s,body)
ws.set_column("A:A",60)
ws.merge_range("A18:H18","주의",band)
for i,s in enumerate(["119 개별 주소는 포함하지 않고 행정동 집계값만 포함합니다.","M9는 물리 감수성만 학습하고 인구·건물·펌프장 변수는 M5 결합 단계에서만 사용합니다.","강서구는 기록 편향으로 물리 지표 동일가중 감수성을 적용합니다."],18): ws.merge_range(i,0,i,7,s,body)
write_df(wb,"모델입력",input_df,{0:18,1:12,2:13,3:14,4:15})
write_df(wb,"기존결과",results,{0:18,1:13,2:15,3:14,4:14,5:14})
write_df(wb,"행정동순위",dong,{0:14,1:15}); write_df(wb,"119검증",fire,{0:14,1:18}); write_df(wb,"지정구역비교",compare,{0:14,1:18})
used=set(input_cols)|set(result_cols)|set(dong.columns)|set(fire.columns)|set(compare.columns); write_df(wb,"변수정의",defs[defs["변수명"].isin(used)],{0:31,1:25,2:22,3:58,4:16,5:16,6:24}); write_df(wb,"데이터출처",sources,{0:10,1:28,2:28,3:15,4:22,5:52,6:58,7:58})
cfg=json.loads((ROOT/"공공데이터/가공데이터/04_모델/M9_최종설정.json").read_text(encoding="utf-8")); config=pd.DataFrame([["M9 학습피처",f,i+1,"정규화된 물리 감수성 변수"] for i,f in enumerate(feats)]+[["XGBoost",k,v,"M9 최종 설정"] for k,v in cfg["params"].items()]+[["XGBoost","scale_pos_weight",cfg["scale_pos_weight"],"M9 최종 설정"]]+[["M5 가중치",k,v,"백분위 가산 결합"] for k,v in {"감수성":.50,"노출":.35,"대응결핍":.15}.items()],columns=["구분","항목","값","설명"]); write_df(wb,"실행설정",config,{0:18,1:28,2:18,3:32})
wb.close()
print(OUT, OUT.stat().st_size)
