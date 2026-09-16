# -*- coding: utf-8 -*-
"""공모전 제출용 활용데이터 CSV 묶음을 생성한다.

최종 보고서의 M9 침수 감수성 모델, M5 점검 우선순위, 119 보조검증에
실제로 사용된 가공 데이터와 산출물만 선별한다. 원본 공간파일과 폐기·실험
산출물은 포함하지 않는다.
"""

from pathlib import Path
import shutil

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "공공데이터" / "가공데이터"
OUT = ROOT / "공공데이터" / "제출용_활용데이터_CSV"
ZIP = ROOT / "output" / "제출용_활용데이터_CSV.zip"

OUT.mkdir(parents=True, exist_ok=True)
ZIP.parent.mkdir(parents=True, exist_ok=True)

FILES = {
    "source": "01_원천데이터_출처.csv",
    "vars": "02_변수정의.csv",
    "input": "03_모델입력_100m격자.csv",
    "m9": "04_M9_침수감수성_100m격자.csv",
    "m5": "05_M5_최종점검순위_100m격자.csv",
    "dong": "06_M5_행정동별_점검순위.csv",
    "fire": "07_119_보조검증_행정동.csv",
    "compare": "08_지정구역_비교후보.csv",
    "index": "00_제출데이터_목록.csv",
}


def save_csv(df: pd.DataFrame, name: str) -> None:
    """한글 Excel 호환 UTF-8 BOM CSV로 저장한다."""
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig", float_format="%.10g")


# ---------------------------------------------------------------------------
# 1. 최종 모델·MCDA 입력 데이터
# ---------------------------------------------------------------------------
features = pd.read_parquet(PROC / "04_모델" / "features_v4.parquet")

m9_raw = [
    "elev_min", "slope_mean", "tpi", "lowland3_ratio", "fluv_area_ratio",
    "rain_annmax_mm", "flow_acc_log", "twi", "dist_stream_m",
    "imperv_ratio", "agri_ratio", "paddy_ratio", "forest_ratio",
    "water_ratio", "road_ratio", "mh_no_dredge_ratio",
]
exposure_raw = [
    "pop", "pop_65", "old_ratio", "bldg_cnt", "floor_area",
    "resid_floor_area", "basement_bldg", "max_floors",
    "underpass_n_300m", "underpass_len_m",
]
capacity_raw = ["pump_dist_m", "pump_n_1000m"]

input_cols = [
    "grid_id", "sgg_cd", "sgg_nm", "adm_cd", "adm_nm", "x_cen", "y_cen",
    "trace_flag", "trace_count", "trace_last_year", "trace_max_depth",
    "hazdist_flag", "hazdist_flood_active",
]
input_cols += m9_raw + [f"{c}_s" for c in m9_raw]
input_cols += exposure_raw + [f"{c}_s" for c in exposure_raw]
input_cols += capacity_raw + [f"{c}_s" for c in capacity_raw]
input_cols = list(dict.fromkeys(input_cols))

missing = [c for c in input_cols if c not in features.columns]
if missing:
    raise KeyError(f"features_v4.parquet에 필요한 열이 없습니다: {missing}")

model_input = features[input_cols].copy()
save_csv(model_input, FILES["input"])


# ---------------------------------------------------------------------------
# 2. 최종 산출물 및 보조검증
# ---------------------------------------------------------------------------
m9 = pd.read_parquet(PROC / "04_모델" / "hazard_score.parquet")
m5 = pd.read_parquet(PROC / "05_산출" / "격자_우선순위.parquet")
dong = pd.read_csv(PROC / "05_산출" / "행정동_우선순위.csv", encoding="utf-8-sig")
fire = pd.read_csv(PROC / "05_산출" / "119검증_행정동.csv", encoding="utf-8-sig")
compare = pd.read_csv(PROC / "05_산출" / "신규완화_구역.csv", encoding="utf-8-sig")

save_csv(m9, FILES["m9"])
save_csv(m5, FILES["m5"])
save_csv(dong, FILES["dong"])
save_csv(fire, FILES["fire"])
save_csv(compare, FILES["compare"])


# ---------------------------------------------------------------------------
# 3. 원천데이터 출처 명세
# ---------------------------------------------------------------------------
sources = pd.DataFrame([
    [1, "행정구역 경계", "국가데이터처 SGIS", "SHP", "2025년 2분기", "100m 격자 행정구역 코드·명칭 부여", "https://www.data.go.kr/data/15129688/fileData.do", "EPSG:5179에서 5186으로 재투영"],
    [2, "SGIS 1km 격자 인구", "국가데이터처 SGIS", "CSV/SHP", "2024년", "인구·고령인구를 건물 연면적 기준으로 100m 격자에 배분", "https://www.data.go.kr/data/15141768/fileData.do", "비밀보호기법 적용 통계로 소규모 값은 근사치"],
    [3, "GIS건물통합정보", "국토교통부·브이월드", "SHP", "2026-08-09 기준", "건물 수·연면적·주거연면적·지하층·최대층수 산출", "https://www.vworld.kr/dtmk/dtmk_ntads_s002.do", "부산 건물 자료를 100m 격자로 집계"],
    [4, "홍수위험지도", "환경부·홍수위험지도 정보제공포털", "SHP", "하천 200년/도시 100년 재현기간", "하천범람 면적비 및 비교 지표 산출", "https://data.floodmap.go.kr/", "도시침수지도는 부산 일부만 구축"],
    [5, "침수흔적도", "행정안전부 재난안전데이터 공유플랫폼", "GeoJSON", "2009~2022년", "지도학습 라벨(trace_flag)과 검증값 산출", "https://www.safetydata.go.kr/disaster-data/view?dataSn=108", "부산 479개 폴리곤을 100m 격자와 교차"],
    [6, "Copernicus DEM GLO-30", "ESA/Airbus", "GeoTIFF", "정적 지형자료", "표고·경사·TPI·저지대·흐름누적·TWI·수계거리 산출", "https://registry.opendata.aws/copernicus-dem/", "30m DEM, 부산 육지 클립 후 EPSG:5186 재투영"],
    [7, "ASOS/AWS 강우 관측", "기상청 기상자료개방포털", "CSV", "ASOS 1990~2026/AWS 2005~2026", "연최대 강우와 119 검증 대상 강우일 산출", "https://data.kma.go.kr/", "부산 및 인근 관측지점 사용"],
    [8, "전국배수펌프장 표준데이터", "지방자치단체·공공데이터포털", "CSV", "수집일 2026-09-03", "펌프장 거리와 1km 내 개수 산출", "https://www.data.go.kr/data/15129436/standard.do", "배수능력 자료 부재로 접근성 대리지표 사용"],
    [9, "자연재해위험개선지구", "행정안전부 재난안전데이터 공유플랫폼", "OpenAPI JSON", "수집일 2026-09-03", "기존 지정구역과 최종 우선순위 비교", "https://www.safetydata.go.kr/disaster-data/view?dataSn=52", "모델 피처가 아닌 비교 기준"],
    [10, "부산광역시 지하차도 현황", "부산광역시·공공데이터포털", "CSV", "수집일 2026-09-03", "300m 내 지하차도 수·길이 산출", "https://www.data.go.kr/data/15119688/fileData.do", "시설명 기반 지오코딩 후 격자 집계"],
    [11, "세분류 토지피복지도 11차", "기후에너지환경부 환경공간정보서비스", "WFS/GPKG", "2021년", "불투수·농경·논·산림·수역·도로 면적비 산출", "https://aid.mcee.go.kr/", "부산 영역을 WFS로 수집하여 100m 격자 집계"],
    [12, "부산 하수맨홀", "부산광역시", "CSV", "2025-12-16 표기 원본", "행정동별 준설 미실시 비율 산출", "https://www.data.go.kr/data/15084501/fileData.do", "좌표가 없어 행정동 단위 비율을 격자에 결합"],
    [13, "부산 119 소방출동정보", "부산광역시·공공데이터포털(Big-데이터웨이브 경유 활용)", "OpenAPI JSON", "2020~2025년 강우일", "배수·풍수해 출동을 행정동 단위 보조검증에 사용", "https://apis.data.go.kr/6260000/Busan119InfoService/getTodayInfo", "개별 주소는 제출자료에서 제외하고 행정동 집계값만 포함"],
], columns=["번호", "데이터명", "제공기관", "원본형식", "대상기간_기준일", "최종분석_사용내용", "출처주소", "비고"])
save_csv(sources, FILES["source"])


# ---------------------------------------------------------------------------
# 4. 변수 정의서
# ---------------------------------------------------------------------------
definitions = {
    "grid_id": ("격자 식별자", "100m 분석격자의 고유 식별자", "문자"),
    "sgg_cd": ("시군구 코드", "시군구 행정구역 코드", "문자"),
    "sgg_nm": ("시군구명", "부산광역시 구·군 명칭", "문자"),
    "adm_cd": ("행정동 코드", "행정동 행정구역 코드", "문자"),
    "adm_nm": ("행정동명", "행정동 명칭", "문자"),
    "x_cen": ("격자 중심 X", "EPSG:5186 좌표계의 격자 중심 X좌표", "m"),
    "y_cen": ("격자 중심 Y", "EPSG:5186 좌표계의 격자 중심 Y좌표", "m"),
    "trace_flag": ("침수이력 라벨", "침수흔적 폴리곤과 교차한 격자 여부(1/0)", "이진"),
    "trace_count": ("침수흔적 수", "격자와 교차한 침수흔적 건수", "건"),
    "trace_last_year": ("최근 침수연도", "격자에서 확인된 가장 최근 침수흔적 연도", "연도"),
    "trace_max_depth": ("최대 침수심", "격자에서 확인된 침수흔적 최대 수심", "m"),
    "hazdist_flag": ("재해위험지구 여부", "전체 유형 자연재해위험개선지구 해당 여부", "이진"),
    "hazdist_flood_active": ("유효 침수유형 지정 여부", "현재 유효한 침수 유형 자연재해위험개선지구 해당 여부", "이진"),
    "elev_min": ("최저표고", "격자 내 최소 표고", "m"),
    "slope_mean": ("평균경사", "격자 내 평균 경사", "degree"),
    "tpi": ("지형위치지수", "주변 지형 대비 상대 고도", "지수"),
    "lowland3_ratio": ("3m 이하 저지대 비율", "격자 내 표고 3m 이하 면적 비율", "0~1"),
    "fluv_area_ratio": ("하천범람 면적비", "격자 내 하천범람 위험지도 중첩 면적 비율", "0~1"),
    "rain_annmax_mm": ("연최대 강우", "관측소별 연최대 강우를 공간보간한 값", "mm"),
    "flow_acc_log": ("흐름누적량", "DEM 흐름누적량의 로그 변환값", "log 지수"),
    "twi": ("지형습윤지수", "경사와 흐름누적량으로 산출한 습윤지수", "지수"),
    "dist_stream_m": ("수계거리", "DEM 기반 수계까지의 거리", "m"),
    "imperv_ratio": ("불투수면 비율", "시가화·건조지역 면적 비율", "0~1"),
    "agri_ratio": ("농업지역 비율", "농업지역 면적 비율", "0~1"),
    "paddy_ratio": ("논 비율", "논 토지피복 면적 비율", "0~1"),
    "forest_ratio": ("산림 비율", "산림지역 면적 비율", "0~1"),
    "water_ratio": ("수역 비율", "수역 면적 비율", "0~1"),
    "road_ratio": ("도로 비율", "도로 토지피복 면적 비율", "0~1"),
    "mh_no_dredge_ratio": ("준설 미실시 비율", "행정동 내 등록 맨홀 중 최종준설일자가 없는 비율", "0~1"),
    "pop": ("인구", "건물 연면적으로 배분한 격자 추정 상주인구", "명"),
    "pop_65": ("고령인구", "격자 추정 65세 이상 인구", "명"),
    "old_ratio": ("고령인구 비율", "격자 인구 중 65세 이상 비율", "0~1"),
    "bldg_cnt": ("건물 수", "격자와 중첩한 건물 수", "동"),
    "floor_area": ("연면적", "격자 내 건물 연면적 합계", "m²"),
    "resid_floor_area": ("주거 연면적", "격자 내 주거용 건물 연면적 합계", "m²"),
    "basement_bldg": ("지하층 건물 수", "지하층이 있는 건물 수", "동"),
    "max_floors": ("최대 지상층수", "격자 내 건물의 최대 지상층수", "층"),
    "underpass_n_300m": ("300m 내 지하차도 수", "격자 중심 300m 이내 지하차도 수", "개"),
    "underpass_len_m": ("지하차도 길이", "연계된 지하차도 총길이", "m"),
    "pump_dist_m": ("배수펌프장 거리", "가장 가까운 배수펌프장까지 거리", "m"),
    "pump_n_1000m": ("1km 내 펌프장 수", "격자 중심 1km 이내 배수펌프장 수", "개"),
    "hazard_raw": ("M9 원점수", "전체 자료로 학습한 XGBoost 침수 감수성 예측확률", "0~1"),
    "hazard_oof": ("M9 공간검증 점수", "자치구 GroupKFold의 OOF 침수 감수성 예측확률", "0~1"),
    "hazard_pct": ("M9 감수성 백분위", "전체 81,224개 격자로 최종 학습한 hazard_raw의 부산 전체 백분위. M5 결합에 사용하며 OOF 점수가 아님", "0~1"),
    "hazard_mcda_pct": ("물리 MCDA 백분위", "16개 물리 지표 동일가중 점수의 백분위", "0~1"),
    "exposure_pct": ("노출 백분위", "인구·건물·지하공간 지표 평균의 백분위", "0~1"),
    "capacity_pct": ("대응결핍 백분위", "배수펌프장 접근성 지표 평균의 백분위", "0~1"),
    "priority": ("최종 점검점수", "감수성 0.50·노출 0.35·대응결핍 0.15 결합값", "0~1"),
    "priority_pct": ("최종 점검순위 백분위", "최종 점검점수의 부산 전체 백분위", "0~1"),
    "priority_src": ("감수성 적용방식", "ML 또는 강서구 물리 MCDA 적용 구분", "문자"),
    "base": ("행정동 통합명", "법정동·행정동 명칭을 결합하기 위한 정규화 명칭", "문자"),
    "dispatch": ("119 배수출동", "강우일의 배수·풍수해 출동 집계", "건"),
    "격자수": ("행정동 격자 수", "행정동에 포함된 100m 분석격자 수", "개"),
    "상위10p격자": ("상위 10% 격자 수", "최종 점검순위 상위 10%에 포함된 격자 수", "개"),
    "평균우선순위": ("평균 점검점수", "해당 행정동 또는 후보 격자의 평균 최종 점검점수", "0~1"),
    "인구": ("집계 인구", "해당 공간단위의 추정 상주인구 합계", "명"),
    "고령인구": ("집계 고령인구", "해당 공간단위의 추정 65세 이상 인구 합계", "명"),
    "지하차도": ("지하차도 수", "해당 공간단위에서 확인된 지하차도 수", "개"),
    "침수흔적격자": ("침수이력 격자 수", "침수흔적 라벨이 1인 100m 격자 수", "개"),
    "재해지정": ("재해위험지구 지정", "유효 침수유형 자연재해위험개선지구 포함 여부", "이진"),
    "상위10p비율": ("상위 10% 격자 비율", "행정동 전체 격자 중 최종순위 상위 10% 격자 비율", "0~1"),
    "위험인구": ("점검대상 위험인구", "상위 10% 격자 비율과 행정동 인구를 곱한 정책용 지표", "명 상당"),
    "순위": ("행정동 점검순위", "위험인구를 기준으로 내림차순 부여한 행정동 순위", "순위"),
    "sgg": ("시군구명", "119 검증용 부산광역시 구·군 명칭", "문자"),
    "격자": ("후보 격자 수", "해당 행정동 또는 통합 행정동의 100m 격자 수", "개"),
    "침수흔적": ("침수이력 격자 수", "침수흔적 라벨이 1인 100m 격자 수", "개"),
    "M5우선순위": ("M5 평균 순위백분위", "통합 행정동 내 M5 최종 점검순위 백분위 평균", "0~1"),
    "M9위험도": ("M9 평균 순위백분위", "통합 행정동 내 M9 OOF 감수성 순위백분위 평균", "0~1"),
    "M5상위10비율": ("M5 상위 10% 비율", "통합 행정동 격자 중 M5 상위 10% 격자 비율", "0~1"),
    "M9상위10비율": ("M9 상위 10% 비율", "통합 행정동 격자 중 M9 상위 10% 격자 비율", "0~1"),
    "출동순위": ("119 출동순위", "119 배수·풍수해 출동 건수의 내림차순 순위", "순위"),
    "M5순위": ("M5 행정동 순위", "M5 평균 순위백분위의 내림차순 순위", "순위"),
    "구분": ("비교 후보 구분", "신규편입권고 또는 완화검토 구분", "문자"),
}

for c in list(definitions):
    if c.endswith("_s"):
        continue
for c in m9_raw + exposure_raw + capacity_raw:
    if c in definitions:
        kname, desc, _unit = definitions[c]
        definitions[f"{c}_s"] = (f"{kname} 정규화", f"{desc}를 위험 증가 방향으로 0~1 정규화한 값", "0~1")

datasets = {
    FILES["input"]: model_input,
    FILES["m9"]: m9,
    FILES["m5"]: m5,
    FILES["dong"]: dong,
    FILES["fire"]: fire,
    FILES["compare"]: compare,
}


identifier_cols = {"grid_id", "sgg_cd", "sgg_nm", "adm_cd", "adm_nm", "x_cen", "y_cen", "sgg", "base"}
label_meta_cols = {"trace_count", "trace_last_year", "trace_max_depth"}
baseline_cols = {"hazdist_flag", "hazdist_flood_active", "재해지정"}
m9_scaled = {f"{c}_s" for c in m9_raw}
m9_original = set(m9_raw)
m5_scaled = {f"{c}_s" for c in exposure_raw + capacity_raw}
m5_original = set(exposure_raw + capacity_raw)


def variable_role(filename: str, col: str) -> str:
    """파일과 변수에 따라 분석 단계의 역할을 명시한다."""
    if col in identifier_cols:
        return "식별자"
    if filename == FILES["input"]:
        if col == "trace_flag":
            return "지도학습 타깃"
        if col in label_meta_cols:
            return "라벨 보조정보·민감도 분석"
        if col in baseline_cols:
            return "기존 지정구역 비교기준"
        if col in m9_scaled:
            return "M9 학습피처"
        if col in m5_scaled:
            return "M5 결합용"
        if col in m9_original or col in m5_original:
            return "전처리 원값·해석용"
    if filename == FILES["m9"]:
        return {
            "hazard_raw": "M9 전체학습 산출",
            "hazard_oof": "M9 공간교차검증·성능평가",
            "hazard_pct": "M5 결합용",
        }.get(col, "M9 산출")
    if filename == FILES["m5"]:
        if col in {"hazard_pct", "hazard_mcda_pct", "exposure_pct", "capacity_pct"}:
            return "M5 결합용"
        if col == "trace_flag":
            return "결과 검증용"
        if col in baseline_cols:
            return "기존 지정구역 비교기준"
        if col == "pop":
            return "정책 해석·집계용"
        return "M5 최종 산출"
    if filename == FILES["dong"]:
        if col in {"trace_flag", "침수흔적격자"}:
            return "결과 검증용"
        if col in baseline_cols:
            return "기존 지정구역 비교기준"
        return "M5 행정동 집계"
    if filename == FILES["fire"]:
        return "119 독립 보조검증"
    if filename == FILES["compare"]:
        return "기존 지정구역 비교 산출"
    return "산출·명세"


var_rows = []
for filename, df in datasets.items():
    for col in df.columns:
        kname, desc, unit = definitions.get(col, (col, "산출 코드에서 생성된 집계 또는 순위 변수", "값"))
        var_rows.append([filename, col, kname, desc, unit, str(df[col].dtype), variable_role(filename, col)])
var_table = pd.DataFrame(var_rows, columns=["파일명", "변수명", "한글명", "정의", "단위_범위", "자료형", "역할"])
save_csv(var_table, FILES["vars"])


# ---------------------------------------------------------------------------
# 5. 제출 파일 목록(자기 자신을 포함한 9개 CSV)
# ---------------------------------------------------------------------------
purpose = {
    FILES["source"]: "최종 분석에 사용한 원천데이터 13종의 제공기관·기간·출처·용도",
    FILES["vars"]: "제출 CSV의 변수명·정의·단위·자료형",
    FILES["input"]: "M9 지도학습과 M5 결합에 실제 사용된 100m 격자 입력값",
    FILES["m9"]: "M9 XGBoost 침수 감수성 원점수·공간교차검증 점수·백분위",
    FILES["m5"]: "M5 최종 점검 우선순위와 감수성·노출·대응결핍 구성점수",
    FILES["dong"]: "행정동별 상위 10% 격자 및 위험인구 기반 점검순위",
    FILES["fire"]: "개별 주소를 제외한 행정동 단위 119 배수·풍수해 출동 보조검증",
    FILES["compare"]: "기존 자연재해위험개선지구와 최종 순위의 비교 후보",
    FILES["index"]: "제출용 활용데이터 파일 구성과 범위 안내",
}
stage = {
    FILES["source"]: "명세", FILES["vars"]: "명세", FILES["input"]: "전처리·모델입력",
    FILES["m9"]: "M9 산출", FILES["m5"]: "M5 산출", FILES["dong"]: "M5 정책집계",
    FILES["fire"]: "독립 보조검증", FILES["compare"]: "기존 지정구역 비교", FILES["index"]: "명세",
}

index_rows = []
ordered = [FILES[k] for k in ["index", "source", "vars", "input", "m9", "m5", "dong", "fire", "compare"]]
for filename in ordered:
    if filename == FILES["index"]:
        rows, cols = len(ordered), 7
    else:
        df = pd.read_csv(OUT / filename, encoding="utf-8-sig", low_memory=False)
        rows, cols = df.shape
    index_rows.append([
        filename, stage[filename], purpose[filename], rows, cols, "UTF-8 with BOM",
        "원본 공간 geometry는 제외하고 EPSG:5186 격자 중심좌표와 행정구역 코드로 제공" if filename == FILES["input"] else "",
    ])
index_df = pd.DataFrame(index_rows, columns=["파일명", "구분", "내용", "행수", "열수", "인코딩", "비고"])
save_csv(index_df, FILES["index"])


# ---------------------------------------------------------------------------
# 6. 무결성 검사 및 ZIP 생성
# ---------------------------------------------------------------------------
assert len(model_input) == 81224
assert model_input["grid_id"].is_unique
assert int(model_input["trace_flag"].sum()) == 2893
assert len(m9) == len(model_input) == len(m5)
assert set(model_input["grid_id"]) == set(m9["grid_id"]) == set(m5["grid_id"])
assert set(m9_raw_i + "_s" for m9_raw_i in m9_raw).issubset(model_input.columns)

for filename in ordered:
    parsed = pd.read_csv(OUT / filename, encoding="utf-8-sig", low_memory=False)
    if parsed.columns.duplicated().any():
        raise AssertionError(f"중복 열 이름: {filename}")

if ZIP.exists():
    ZIP.unlink()
shutil.make_archive(str(ZIP.with_suffix("")), "zip", root_dir=OUT.parent, base_dir=OUT.name)

print(f"생성 폴더: {OUT}")
print(f"생성 ZIP: {ZIP}")
for filename in ordered:
    d = pd.read_csv(OUT / filename, encoding="utf-8-sig", low_memory=False)
    print(f"- {filename}: {len(d):,}행 × {len(d.columns)}열")
