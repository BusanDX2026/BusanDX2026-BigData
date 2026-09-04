# -*- coding: utf-8 -*-
"""
S12. 부산 119 소방출동정보 수집 (강우일 한정)

■ 왜 필요한가 (이슈 #29 대응)
  침수흔적은 사건이 18개뿐이고 자치구에 쏠려 있다(2012-09·2019-07 = 100% 강서).
  자치구 GroupKFold 로 검증하면 그 사건 자체가 학습에서 사라져, 활성화 강우 모델의
  구조적 상한(분리 AUC 0.688)이 생겼다. 119 출동은 **전 자치구에서 매일** 발생하므로
  사건 수를 크게 늘릴 수 있다.

■ 확인된 API 사양 (2026-09-04 탐침)
  END   : https://apis.data.go.kr/6260000/Busan119InfoService/getTodayInfo
  총건수 : 4,096,747 (2013-08 ~ 2026, 연 18~26만건, 결측 없음)
  파라미터: regtime (접두 매칭 — 20140825 / 202007 / 2020 모두 가능)
  필드  : regtime, dsraddr(재난지점·법정동), dsrkndcd(종별), dsrclscd(구분),
          dsrsizecd(규모), juriswardid1(관할서), juriswardid2(안전센터)

■ 중요한 한계 — 침수 신호는 2020년경부터만
  2020-07-23: 자연재해 193 · 지원출동(배수) 129  → 일평균 대비 1.6배 급증
  2014-08-25: 해당 코드 자체가 없음(분류 19종) → 일평균 대비 1.0배, 신호 없음
  분류 체계가 2020년경 바뀌며 침수 출동이 비로소 구분되기 시작했다.
  따라서 **2020년 이후만** 침수 신호로 사용한다.

■ 수집 범위 — 전량이 아니라 강우일만
  전체 2020~2026 = 150만건(API 3,000회+). 대신 이미 보유한 AWS 강우로
  '전지점 최대 3시간 강우 >= 30mm' 인 날만 고른다 → 100일 (~200회 호출).

■ 입출력
  IN : 02_레이어별/강우_AWS시간_long.parquet, secrets/data_go_kr_key.txt
  OUT: 02_레이어별/119출동_강우일_원시.parquet, _리포트/S12_119출동수집.md
■ 규약: 인증키는 코드에 두지 않음 (secrets/ 또는 환경변수 DATA_GO_KR_KEY)
"""
import sys, io, os, json, time, urllib.request, urllib.parse, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"
LAY = GG / "02_레이어별"; REP = GG / "_리포트"
BASE = "https://apis.data.go.kr/6260000/Busan119InfoService/getTodayInfo"
RAIN_TH = 30.0            # 전지점 최대 3시간 강우 임계 (mm)
START, END = "2020-01-01", "2026-09-05"
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

def get_key():
    k = os.environ.get("DATA_GO_KR_KEY")
    if k:
        return k.strip()
    f = ROOT / "secrets" / "data_go_kr_key.txt"
    if f.exists():
        return f.read_text(encoding="utf-8").strip()
    raise RuntimeError("공공데이터포털 키 없음: 환경변수 DATA_GO_KR_KEY 또는 secrets/data_go_kr_key.txt 필요")

KEY = get_key()

P("# S12. 부산 119 소방출동정보 수집 (강우일 한정)")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

# ---------------------------------------------------------------
# 1. 대상 강우일 선정
# ---------------------------------------------------------------
P("## 1. 대상 강우일 선정")
rain = pd.read_parquet(LAY / "강우_AWS시간_long.parquet")
rain["tm"] = pd.to_datetime(rain.tm)
r = rain[(rain.tm >= START) & (rain.tm < END)]
piv = r.pivot_table(index="tm", columns="stn", values="rain_mm", aggfunc="max").sort_index().asfreq("h").fillna(0)
r3 = piv.rolling(3, min_periods=1).sum().max(axis=1)
day3 = r3.groupby(r3.index.normalize()).max()
days = day3[day3 >= RAIN_TH].index
P(f"- 기간 {START} ~ {END}, 전지점 최대3h >= {RAIN_TH:.0f}mm 인 날: **{len(days)}일**")
P(f"- 강우 분포(대상일) p25/50/75/max = "
  f"{day3[days].quantile(.25):.0f}/{day3[days].median():.0f}/{day3[days].quantile(.75):.0f}/{day3[days].max():.0f} mm")

# ---------------------------------------------------------------
# 2. API 수집
# ---------------------------------------------------------------
P("\n## 2. API 수집")
def fetch_day(yyyymmdd, page_size=500, max_pages=40, retry=3):
    out, page = [], 1
    while page <= max_pages:
        q = urllib.parse.urlencode({"serviceKey": KEY, "resultType": "json",
                                    "regtime": yyyymmdd, "numOfRows": page_size, "pageNo": page})
        body = None
        for a in range(retry):
            try:
                req = urllib.request.Request(f"{BASE}?{q}", headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=40) as resp:
                    body = json.loads(resp.read().decode("utf-8", "replace"))
                break
            except Exception:
                time.sleep(1.5 * (a + 1))
        if body is None:
            return out, False
        try:
            it = body["response"]["body"]["items"]["item"]
        except Exception:
            break
        it = it if isinstance(it, list) else [it]
        if not it:
            break
        out += it
        if len(it) < page_size:
            break
        page += 1
    return out, True

rows, failed = [], []
t0 = time.perf_counter()
for i, d in enumerate(days, 1):
    ymd = f"{d:%Y%m%d}"
    got, ok = fetch_day(ymd)
    if not ok:
        failed.append(ymd)
    rows += got
    if i % 20 == 0 or i == len(days):
        P(f"  진행 {i}/{len(days)}일 · 누적 {len(rows):,}건 · {time.perf_counter()-t0:.0f}s")
df = pd.DataFrame(rows)
P(f"- 수집 완료: **{len(df):,}건** / {len(days)}일 ({time.perf_counter()-t0:.0f}s)")
if failed:
    P(f"- ⚠ 조회 실패일 {len(failed)}건: {failed[:10]}")

df["regtime"] = df["regtime"].astype(str)
df["dt"] = pd.to_datetime(df.regtime, format="%Y%m%d%H%M%S", errors="coerce")
df["day"] = df.dt.dt.normalize()
df = df.dropna(subset=["dt"])
df["juriswardid2"] = df["juriswardid2"].astype(str).str.replace("!N!", "", regex=False).str.strip()

# ---------------------------------------------------------------
# 3. 침수 관련 출동 분류
# ---------------------------------------------------------------
P("\n## 3. 침수 관련 출동 분류")
FLOOD = ["자연재해", "지원출동(배수)", "지원출동(풍수해)"]
df["is_flood"] = df.dsrclscd.isin(FLOOD).astype(int)
P(f"- 침수 관련 코드: {FLOOD}")
P(f"- 침수 관련 출동 **{int(df.is_flood.sum()):,}건** / 전체 {len(df):,}건 ({df.is_flood.mean():.1%})")
P("\n연도별 (분류 체계 변화 확인 — 2020년 이전은 코드 자체가 없음)")
P("| 연도 | 대상일 | 전체출동 | 침수출동 | 침수비율 |")
P("|--:|--:|--:|--:|--:|")
for y, g in df.groupby(df.dt.dt.year):
    P(f"| {y} | {g.day.nunique()} | {len(g):,} | {int(g.is_flood.sum()):,} | {g.is_flood.mean():.1%} |")

P("\n침수 출동 상위 일자")
top = df[df.is_flood == 1].groupby("day").size().sort_values(ascending=False).head(10)
P("| 일자 | 침수출동 | 그날 최대3h강우 |")
P("|---|--:|--:|")
for d, c in top.items():
    P(f"| {d:%Y-%m-%d} | {c:,} | {day3.get(d, float('nan')):.0f} mm |")

P("\n구분(dsrclscd) 상위 15 — 전체")
for k, v in df.dsrclscd.value_counts().head(15).items():
    P(f"  {v:6,}  {k}")

df.to_parquet(LAY / "119출동_강우일_원시.parquet", index=False)
P(f"\n저장: {LAY/'119출동_강우일_원시.parquet'} ({df.shape})")
P(f"- dsraddr 고유값 {df.dsraddr.nunique():,}개 (법정동 단위) → 행정동 매칭은 S13에서")

(REP / "S12_119출동수집.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'S12_119출동수집.md'}", flush=True)
