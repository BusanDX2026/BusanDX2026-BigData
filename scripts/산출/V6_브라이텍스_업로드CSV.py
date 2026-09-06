# -*- coding: utf-8 -*-
"""
V6. Big-데이터웨이브 셀프분석(Brightics AI) 업로드용 CSV 생성

■ 용도
  대회 서식 붙임 4-2 "big-데이터웨이브 데이터명 및 브라이텍스 사용결과 캡처".
  셀프분석은 카탈로그 연동이 없고 파일 업로드만 지원하므로(2026-09-06 확인),
  S12 가 API 로 수집한 원시 레코드를 CSV 로 내보낸다.

■ 원칙 — 집계하지 않는다
  로컬에서 집계까지 끝낸 결과를 올리면 "분석은 밖에서 끝나고 도구는 그림만 그린" 셈이라
  증빙이 되지 않는다. **원시 레코드**만 올리고 Group By 는 Brightics 안에서 수행한다.
  날짜 필터링(강우일 100일)은 집계가 아니라 수집 범위이므로 그대로 둔다.

■ 데이터 출처
  Big-데이터웨이브(data.busan.go.kr/bdip) `부산광역시_119 소방출동정보` API
  수집 스크립트: scripts/전처리/S12_119출동_수집.py

■ Brightics 에서 확인될 수치 (MS5 리포트와 일치해야 함)
  자연재해 6,383 / 지원출동(배수) 570 / 지원출동(풍수해) 464
  → 침수 신호 = 570 + 464 = 1,034건

- 입력: 02_레이어별/119출동_강우일_원시.parquet
- 출력: 06_제출자료/붙임4_119출동_브라이텍스업로드.csv
"""
import sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"
OUT = GG / "06_제출자료"; OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(GG / "02_레이어별" / "119출동_강우일_원시.parquet")
print(f"원시 {len(df):,}건")

# S12 코드리뷰 반영: API 의 regtime 은 접두가 아니라 **부분문자열** 매칭이라
#   '20200903' 조회에 '20041120200903'(2004-11-20) 이 딸려 온다.
#   MS5 와 동일하게 '대상 강우일 집합'에 속하는지로 걸러야 140건이 제거된다.
RAIN_TH = 30.0   # MS5 와 동일: 전지점 최대 3시간 누적 30mm 이상인 날
rain = pd.read_parquet(GG / "02_레이어별" / "강우_AWS시간_long.parquet")
rain["tm"] = pd.to_datetime(rain.tm)
r = rain[(rain.tm >= "2020-01-01") & (rain.tm < "2026-09-05")]
piv = r.pivot_table(index="tm", columns="stn", values="rain_mm", aggfunc="max").sort_index().asfreq("h").fillna(0)
day3 = piv.rolling(3, min_periods=1).sum().max(axis=1).pipe(lambda s: s.groupby(s.index.normalize()).max())
days = set(d.strftime("%Y%m%d") for d in day3[day3 >= RAIN_TH].index)

ymd = df.regtime.astype(str).str[:8]
n_raw = len(df)
df = df[ymd.isin(days)].copy()
print(f"대상 강우일({len(days)}일) 소속 {len(df):,}건 "
      f"— 부분문자열 오염 {n_raw-len(df)}건 제거")

# Brightics 는 Spark CSV 파서라 한글 컬럼명·타임스탬프에서 걸리기 쉽다.
#   컬럼명은 원본 API 필드명 유지(영문), 날짜는 문자열로 고정.
out = pd.DataFrame({
    "regtime":   df.regtime.astype(str),          # 접수시각 (원본)
    "dt":        df.dt.dt.strftime("%Y-%m-%d %H:%M:%S"),
    "day":       df.day.astype(str),              # 강우일
    "dsraddr":   df.dsraddr.astype(str),          # 주소 (동 단위, 개인정보 없음)
    "dsrkndcd":  df.dsrkndcd.astype(str),         # 출동종별 (구급/구조/화재/기타)
    "dsrclscd":  df.dsrclscd.astype(str),         # 출동분류  ← Group By 대상
    "dsrsizecd": df.dsrsizecd.astype(str),        # 출동규모
    "juriswardid1": df.juriswardid1.astype(str),  # 관할소방서
    "juriswardid2": df.juriswardid2.astype(str),  # 관할안전센터
})

# ── Brightics CSV 파서 대응 ───────────────────────────────────
# Brightics(AppendableParquetWriter)는 RFC4180 을 따르지 않는다. 따옴표로 감싼 필드를
#   해석하지 못하고 그냥 쉼표로 자르므로, 값 안에 쉼표가 있으면 열 개수가 안 맞아
#   "Cannot upload file, please check row data" 로 업로드가 통째로 실패한다.
#   실제 사례: dsrclscd = '고층건물(3층이상,아파' (2026-09-06 업로드 379행에서 실패)
# → 값 안의 쉼표·따옴표·개행을 미리 제거하고 **따옴표 없이(QUOTE_NONE)** 기록한다.
#   대상은 dsrclscd 3종 389행뿐이고, 집계 대상인 자연재해·지원출동(배수)·(풍수해)는
#   영향받지 않는다.
import csv as _csv
bad_cells = 0
for c in out.columns:
    s = out[c].astype(str)
    n = int((s.str.contains(",", regex=False) | s.str.contains('"', regex=False)).sum())
    if n:
        print(f"  정제 `{c}` {n}행 — 쉼표→· , 따옴표 제거")
        bad_cells += n
    out[c] = (s.str.replace(",", "·", regex=False)
               .str.replace('"', "", regex=False)
               .str.replace("\r", " ", regex=False)
               .str.replace("\n", " ", regex=False)
               .str.strip())
assert not out.apply(lambda s: s.str.contains(",", regex=False)).any().any(), "쉼표 잔존"

# 파일명·인코딩: 한글 파일명은 업로드에서 깨질 수 있어 ASCII 로. BOM 없는 순수 UTF-8.
p = OUT / "119.csv"
out.to_csv(p, index=False, encoding="utf-8", quoting=_csv.QUOTE_NONE)
mb = p.stat().st_size / 1024 / 1024
print(f"\n저장: {p.name}  ({len(out):,}행 × {out.shape[1]}열, {mb:.1f} MB, 정제 {bad_cells}셀)")

print("\n=== Brightics Group By(dsrclscd) 로 나와야 할 값 ===")
v = out.dsrclscd.value_counts()
for k in ["자연재해", "지원출동(배수)", "지원출동(풍수해)"]:
    print(f"  {k:<14} {v.get(k, 0):>6,}")
flood = int(v.get("지원출동(배수)", 0) + v.get("지원출동(풍수해)", 0))
print(f"  {'침수 신호 합계':<14} {flood:>6,}")
print(f"\n전체 분류 {out.dsrclscd.nunique()}종 · 총 {len(out):,}건")
