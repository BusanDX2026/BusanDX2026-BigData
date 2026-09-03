# -*- coding: utf-8 -*-
"""
S6. 마스터 격자 테이블 결합
- 입력: 01_격자/grid_100m.gpkg + 02_레이어별/*.parquet
- 출력:
    03_마스터/master_grid.parquet (+ .gpkg)  ← 피처만 (target leakage 방지)
    03_마스터/target_grid.parquet             ← 침수흔적 (검증 타깃, 피처 아님)
    03_마스터/baseline_grid.parquet           ← 재해위험지구 (행정지정 대조, 피처 아님)
- 규약: S0 §4(공간미포함=0/미관측=NaN), §9(danger_level·침수흔적 미사용, 흔적/지정은 분리)
"""
import sys, io, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd

ROOT = Path(__file__).resolve().parents[2]   # 저장소 루트 (절대경로 하드코딩 금지)
GG = ROOT / "공공데이터" / "가공데이터"
LYR = GG / "02_레이어별"
OUT = GG / "03_마스터"; OUT.mkdir(parents=True, exist_ok=True)
REP = GG / "_리포트"
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# S6. 마스터 격자 결합")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

grid = gpd.read_file(GG / "01_격자" / "grid_100m.gpkg")
P(f"기준 격자: {len(grid):,}행, 키 grid_id (유일 {grid.grid_id.is_unique})")

# ---- 피처 레이어 (공간미포함=0) ----
FEATURES = {
    "인구건물_grid.parquet":   dict(fill0=["pop","pop_65","bldg_cnt","floor_area","resid_floor_area",
                                          "basement_bldg","max_floors","old_ratio"]),
    "홍수위험_grid.parquet":   dict(fill0=["fluv_grade","fluv_area_ratio"]),  # pluv_*는 NaN 유지
    "지형_grid.parquet":       dict(fill0=[]),                                 # DEM 미관측 NaN 유지
    "배수펌프장_grid.parquet": dict(fill0=["pump_dist_m","pump_n_500m","pump_n_1000m"]),
    "강우_grid.parquet":       dict(fill0=[]),                                 # 전 격자 값 있음
    "지하차도_grid.parquet":   dict(fill0=["underpass_cnt","underpass_len_m",
                                          "underpass_n_300m","underpass_n_500m"]),
}
m = grid[["grid_id","sgg_cd","sgg_nm","adm_cd","adm_nm","x_cen","y_cen","geometry"]].copy()
used = []
for fn, cfg in FEATURES.items():
    p = LYR / fn
    if not p.exists():
        P(f"- (건너뜀, 없음) {fn}")
        continue
    d = pd.read_parquet(p)
    newcols = [c for c in d.columns if c != "grid_id"]
    m = m.merge(d, on="grid_id", how="left")
    for c in cfg["fill0"]:
        if c in m.columns:
            m[c] = m[c].fillna(0)
    used.append(fn)
    P(f"- {fn}: +{len(newcols)}컬럼 {newcols}")

feat_cols = [c for c in m.columns if c not in ("grid_id","sgg_cd","sgg_nm","adm_cd","adm_nm","x_cen","y_cen","geometry")]
P(f"\n피처 컬럼 {len(feat_cols)}개")
na = m[feat_cols].isna().mean().sort_values(ascending=False)
P("결측률 상위:")
for c, v in na[na > 0].items():
    P(f"  {c}: {v:.1%}")

pd.DataFrame(m.drop(columns="geometry")).to_parquet(OUT / "master_grid.parquet", index=False)
m.to_file(OUT / "master_grid.gpkg", driver="GPKG")
P(f"\n저장: 03_마스터/master_grid.parquet ({len(m):,}행 x {len(m.columns)-1}열), master_grid.gpkg")

# ---- 타깃 / 베이스라인 (분리 저장) ----
tp = LYR / "침수흔적_grid.parquet"
if tp.exists():
    t = grid[["grid_id","sgg_cd","adm_cd"]].merge(pd.read_parquet(tp), on="grid_id", how="left")
    t.to_parquet(OUT / "target_grid.parquet", index=False)
    P(f"저장: target_grid.parquet (침수흔적, 피처 아님) — trace_flag=1 {int(t.trace_flag.fillna(0).sum()):,}")
bp = LYR / "재해위험지구_grid.parquet"
if bp.exists():
    b = grid[["grid_id","sgg_cd","adm_cd"]].merge(pd.read_parquet(bp), on="grid_id", how="left")
    b.to_parquet(OUT / "baseline_grid.parquet", index=False)
    P(f"저장: baseline_grid.parquet (재해위험지구, 대조용)")
else:
    P("(재해위험지구_grid.parquet 아직 없음 — S5f 후 재실행)")

P("\n## 검증 (S0 §9) — 위반 시 실행 중단")
leak = [c for c in feat_cols if any(k in c.lower() for k in ["trace", "danger", "hazdist"])]
assert not leak, f"target/baseline leakage: {leak}"
assert len(m) == len(grid), "격자 행수 불일치"
assert m.grid_id.is_unique, "grid_id 중복"
assert m.crs.to_epsg() == 5186, f"CRS {m.crs.to_epsg()} != 5186"
assert m.sgg_cd.isna().sum() == 0 and m.adm_cd.isna().sum() == 0, "행정코드 결측"
# 물리적 범위 불변식
INVARIANTS = [
    ("old_ratio ∈ [0,1]", lambda: m.old_ratio.dropna().between(0, 1).all()),
    ("pop_65 ≤ pop", lambda: (m.pop_65 <= m["pop"] + 1e-6).all()),
    ("resid_floor_area ≤ floor_area", lambda: (m.resid_floor_area <= m.floor_area + 1e-3).all()),
    ("면적비 ∈ [0,1]", lambda: all(m[c].dropna().between(0, 1).all()
                                  for c in ["fluv_area_ratio", "lowland5_ratio", "lowland3_ratio"])),
    ("fluv_grade ∈ [0,5]", lambda: m.fluv_grade.dropna().between(0, 5).all()),
    ("lowland3 ≤ lowland5", lambda: (m.lowland3_ratio.fillna(0) <= m.lowland5_ratio.fillna(0) + 1e-9).all()),
    ("음수 없는 카운트/면적", lambda: all((m[c].dropna() >= 0).all() for c in
                                  ["pop", "pop_65", "bldg_cnt", "floor_area", "pump_dist_m"])),
]
for name, fn in INVARIANTS:
    ok = fn()
    P(f"- {name}: {'OK' if ok else 'FAIL'}")
    assert ok, f"불변식 위반: {name}"
# 완전중복 컬럼 경보 (제거 대상 조기 탐지)
import itertools
numc = m[feat_cols].select_dtypes("number")
dup = [(a, b) for a, b in itertools.combinations(numc.columns, 2)
       if numc[[a, b]].dropna().shape[0] > 100 and abs(numc[[a, b]].dropna().corr().iloc[0, 1]) > 0.9999]
P(f"- 완전중복(|r|>0.9999) 쌍: {dup if dup else '없음'}  → M4에서 1개만 사용")
P(f"- leakage 없음 / 행수 {len(m):,} / CRS {m.crs.to_epsg()} ✓")

(REP / "S6_마스터.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'S6_마스터.md'}", flush=True)
