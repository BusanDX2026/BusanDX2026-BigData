# -*- coding: utf-8 -*-
"""
D2. AI-Hub「부산시 침수위험 복합 데이터」보존 패키지 생성

■ 왜
  원본 39.85GB(JPG 114,763 / JSON 114,763)를 저장소에 계속 두기 어렵다.
  삭제하기 전에 **실제로 쓴 것 + 나중에 판단하는 데 필요한 근거**만 압축해 남긴다.

■ 보존 대상
  1) S7 이 실제로 사용한 것 — 측구 관측지점 19개 (이미 가공데이터에 있음, 여기로 사본)
  2) 수치모델 시나리오 매니페스트 — 구분·지역·재현빈도·지속시간·강우량·이미지수
  3) 샘플 JSON 3개 + 샘플 이미지 2장 (구조 증빙용)
  4) README — 출처·구성·활용 가능성 판단·삭제 사유

■ 출력: 공공데이터/raw/15_AIHub_침수위험복합/
"""
import sys, io, json, os, shutil, glob
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "부산시 침수위험 복합 데이터"
GG = ROOT / "공공데이터" / "가공데이터"
OUT = ROOT / "공공데이터" / "raw" / "15_AIHub_침수위험복합"
(OUT / "샘플").mkdir(parents=True, exist_ok=True)

if not SRC.exists():
    raise SystemExit(f"원본 폴더 없음: {SRC}")

# ── 1. S7 이 사용한 관측지점 사본 ─────────────────────────────
for f in ["aihub_관측지점_원시.csv", "aihub_cctv_매핑.csv"]:
    p = GG / "02_레이어별" / f
    if p.exists():
        shutil.copy2(p, OUT / f)
        print(f"사본: {f}")

# ── 2. 수치모델 시나리오 매니페스트 ───────────────────────────
rows = []
for split, tag in [("Training", "TL"), ("Validation", "VL")]:
    base = SRC / split / "02.라벨링데이터" / tag / "침수위험 수치모델 이미지 데이터"
    if not base.exists():
        continue
    for typ in sorted(os.listdir(base)):
        for area in sorted(os.listdir(base / typ)):
            for scen in sorted(os.listdir(base / typ / area)):
                d = base / typ / area / scen
                js = [f for f in os.listdir(d) if f.endswith(".json")]
                r1 = r24 = None
                nums = []
                for f in js:
                    try:
                        nums.append(int(f.rsplit("_", 1)[-1].split(".")[0]))
                    except Exception:
                        pass
                if js:
                    try:
                        info = json.load(open(d / js[0], encoding="utf-8"))["INFO"]
                        r1, r24 = info.get("1HR_RAINFALL"), info.get("24HR_RAINFALL")
                    except Exception:
                        pass
                rows.append(dict(split=split, 구분=typ, 지역=area, 시나리오=scen,
                                 재현빈도=scen.split("yr")[0], 지속시간_분=scen.split("_")[1] if "_" in scen else "",
                                 이미지수=len(js), 타일번호_최대=max(nums) if nums else None,
                                 강우_1시간mm=r1, 강우_24시간mm=r24))
man = pd.DataFrame(rows)
man.to_csv(OUT / "수치모델_시나리오_매니페스트.csv", index=False, encoding="utf-8-sig")
print(f"매니페스트: {len(man)}행 · 이미지 {man.이미지수.sum():,}장")

# 시나리오 요약 (재현빈도별 강우)
summ = (man.groupby(["재현빈도", "지속시간_분"])
          .agg(이미지수=("이미지수", "sum"), 강우_1시간mm=("강우_1시간mm", "first"),
               강우_24시간mm=("강우_24시간mm", "first")).reset_index())
summ.to_csv(OUT / "수치모델_재현빈도별_강우.csv", index=False, encoding="utf-8-sig")
print(f"재현빈도 요약: {len(summ)}행")

# ── 3. 샘플 (구조 증빙) ───────────────────────────────────────
samp = []
for pat, n in [("**/침수위험 수치모델 이미지 데이터/내수/동래구/030yr_060/*.json", 1),
               ("**/침수위험 수치모델 이미지 데이터/외수/온천천/*/*.json", 1),
               ("**/침수위험 지역 라벨링 데이터/**/*.json", 1)]:
    g = sorted(glob.glob(str(SRC / pat), recursive=True))[:n]
    samp += g
for s in samp:
    shutil.copy2(s, OUT / "샘플" / Path(s).name)
    print(f"샘플 JSON: {Path(s).name}")
for nm in ["Dongnae_030_1_00001", "Dongnae_030_1_00050"]:
    g = glob.glob(str(SRC / f"**/침수위험 수치모델 이미지 데이터/**/{nm}.jpg"), recursive=True)
    if g:
        shutil.copy2(g[0], OUT / "샘플" / f"{nm}.jpg")
        print(f"샘플 이미지: {nm}.jpg")

sz = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
print(f"\n보존 패키지 총 {sz/1024/1024:.1f} MB → {OUT}")
