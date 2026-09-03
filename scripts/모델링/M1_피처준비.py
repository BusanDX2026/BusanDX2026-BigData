# -*- coding: utf-8 -*-
"""
M1. 피처 준비 — 정리·파생·변환·정규화

설계 원칙 (전처리 코드리뷰 결과 반영)
  1) 피처를 개념군으로 분리한다: HAZARD(물리) / EXPOSURE(노출) / CAPACITY(대응)
     └ 이유: 타깃(침수흔적)은 '실제 침수'가 아니라 '기록된 침수'라 인구 많은 곳에 편중(코드리뷰 FIND-3).
        인구·건물을 피처로 넣고 P(침수흔적)을 학습하면 물리적 위험이 아니라 '보고편향'을 학습한다.
        따라서 지도학습은 HAZARD만으로 '침수 감수성'을 학습하고, 노출·대응은 MCDA에서 결합한다.
        (편향 크기는 M2에서 HAZARD-only vs ALL 비교로 정량화)
  2) 완전중복 피처는 쌍당 1개만 (코드리뷰 §6)
  3) 절대표고 저지대(lowland*)는 부산에서 예측력 없음 → 상대표고(tpi) 우선 (FIND-1)
  4) 침수심 등급(fluv_grade)은 속성정의서 미확보 가정이므로 면적비를 주 피처로 (한계 L-1)

- 입력: 03_마스터/master_grid.parquet, target_grid.parquet, baseline_grid.parquet
- 출력: 04_모델/features.parquet (원값+변환값+그룹메타), 04_모델/feature_meta.csv
- 규약: 문서/S0_작업규약.md §6(robust min-max, log1p), §9
"""
import sys, io, json, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"
MST = GG / "03_마스터"
OUT = GG / "04_모델"; OUT.mkdir(parents=True, exist_ok=True)
REP = GG / "_리포트"
RANDOM_STATE = 42
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# M1. 피처 준비")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

m = pd.read_parquet(MST / "master_grid.parquet")
t = pd.read_parquet(MST / "target_grid.parquet")
b = pd.read_parquet(MST / "baseline_grid.parquet")
P(f"입력: master {m.shape}, target {t.shape}, baseline {b.shape}")

# ---------------------------------------------------------------
# 1. 피처 그룹 정의 + 채택/기각 사유
# ---------------------------------------------------------------
GROUPS = {
    # --- HAZARD: 물이 어디로 모이고 넘치는가 (지도학습 입력) ---
    #   자연·물리 지표만. '행정이 이미 위험하다고 판단한 결과물'은 넣지 않는다(순환논리 방지).
    "HAZARD": ["elev_min", "slope_mean", "tpi", "lowland3_ratio",
               "fluv_area_ratio", "rain_annmax_mm"],
    # --- EXPOSURE: 누가/무엇이 피해를 보는가 (MCDA에서 결합, 지도학습 제외) ---
    "EXPOSURE": ["pop", "pop_65", "old_ratio", "bldg_cnt", "floor_area",
                 "resid_floor_area", "basement_bldg", "max_floors",
                 "underpass_n_300m", "underpass_len_m"],
    # --- CAPACITY: 대응 여력 (MCDA 결합, 지도학습 제외) ---
    #   ⚠ 배수펌프장은 '침수 위험한 곳에 설치'되므로 역인과. 침수 예측 피처로 쓰면
    #     재해위험지구를 피처로 쓰는 것과 같은 순환논리가 된다 → HAZARD에서 제외하고 CAPACITY로만.
    "CAPACITY": ["pump_dist_m", "pump_n_1000m"],
}
DROPPED = {
    "pop_density":       "완전중복 pop×100 (전처리에서 이미 제거)",
    "rain_ratio":        "완전중복 r=1.000 with rain_annmax_mm → 절대값 채택",
    "fluv_grade":        "fluv_area_ratio와 r=0.997. 등급-침수심 대응이 속성정의서 미확보 가정(L-1) → 가정 없는 면적비 채택",
    "elev_mean":         "elev_min과 r=0.993. 침수는 셀 내 '최저점'에 고이므로 물리적으로 elev_min 채택",
    "lowland5_ratio":    "lowland3와 r=0.924 중복 + 절대표고 저지대는 부산서 예측력 없음(FIND-1) → 더 엄격한 3m만 보조 유지",
    "pluv_grade":        "강서구만 구축, 격자 76% 결측, 침수흔적 상관 -0.03 → 전역 모델 제외(L-2). 강서 케이스스터디에서만 사용",
    "pluv_area_ratio":   "동상",
    "underpass_cnt":     "격자 0.055%만 비영(분산≈0) → 이웃 피처 underpass_n_300m로 대체(DESIGN-2)",
    "underpass_n_500m":  "underpass_n_300m와 고중복, 300m가 침수 통제 반경에 더 적합",
    "underpass_dist_m":  "지하차도 없는 격자에서 의미 희석(최대 20km) → 이웃 개수로 대체",
    "underpass_min_height_m": "99.9% 결측(지하차도 있는 셀만) → 메타데이터, 피처 아님",
    "pump_n_500m":       "pump_n_1000m와 개념중복, 1km가 펌프 서비스권역에 더 근사",
}
P("\n## 1. 피처 그룹")
for g, cols in GROUPS.items():
    P(f"- **{g}** ({len(cols)}): {cols}")
P(f"\n## 2. 기각 피처 {len(DROPPED)}개")
for c, why in DROPPED.items():
    P(f"- `{c}` — {why}")

USE = GROUPS["HAZARD"] + GROUPS["EXPOSURE"] + GROUPS["CAPACITY"]
missing = [c for c in USE if c not in m.columns]
assert not missing, f"master에 없는 피처: {missing}"

# ---------------------------------------------------------------
# 2. 결측 처리
# ---------------------------------------------------------------
P("\n## 3. 결측 처리")
df = m[["grid_id", "sgg_cd", "sgg_nm", "adm_cd", "adm_nm", "x_cen", "y_cen"] + USE].copy()
n0 = len(df)
# DEM 없는 셀 = 해상/경계밖 → 분석 대상 아님, 제거
no_dem = df.elev_min.isna()
P(f"- DEM 결측 격자 {int(no_dem.sum()):,} 제거 (해상·경계밖, 육지 분석 대상 아님)")
df = df[~no_dem].copy()
# slope는 DEM 가장자리에서 NaN(중심차분) → 자치구 중앙값 대체 + 결측 표시
for c in ["slope_mean", "tpi", "lowland3_ratio"]:
    n_na = int(df[c].isna().sum())
    if n_na:
        df[c] = df[c].fillna(df.groupby("sgg_cd")[c].transform("median")).fillna(df[c].median())
        P(f"- `{c}` 결측 {n_na:,} → 자치구 중앙값 대체 (DEM 가장자리 중심차분 산물)")
assert df[USE].isna().sum().sum() == 0, "잔여 결측"
P(f"- 최종 분석 격자: {len(df):,} (원 {n0:,}, -{n0-len(df):,})")

# ---------------------------------------------------------------
# 3. 타깃 결합
# ---------------------------------------------------------------
df = df.merge(t[["grid_id", "trace_flag", "trace_count", "trace_last_year", "trace_max_depth"]],
              on="grid_id", how="left")
df["trace_flag"] = df.trace_flag.fillna(0).astype(int)
df["trace_count"] = df.trace_count.fillna(0)
df = df.merge(b[["grid_id", "hazdist_flag", "hazdist_flood_active"]], on="grid_id", how="left")
for c in ["hazdist_flag", "hazdist_flood_active"]:
    df[c] = df[c].fillna(0).astype(int)
P(f"\n## 4. 타깃")
P(f"- trace_flag=1: {int(df.trace_flag.sum()):,} / {len(df):,} = **{df.trace_flag.mean():.2%}** (심한 불균형 → PR-AUC 병행 필수)")
P(f"- 2014년 이외 침수 격자: {int(((df.trace_flag==1)&(df.trace_last_year!=2014)).sum()):,} (FIND-2 홀드아웃용)")
P(f"- 재해위험지구(유효+침수) 격자: {int(df.hazdist_flood_active.sum()):,} (baseline 대조용)")

# ---------------------------------------------------------------
# 4. 변환 (S0 §6): 왜도 큰 지표 log1p → robust min-max(5~95%)
# ---------------------------------------------------------------
P("\n## 5. 변환 (log1p → robust min-max 5~95%)")
# 방향 정렬: 위험↑ → 점수↑ 이어야 함. 부호 반전 대상 명시
INVERT = {
    "elev_min":     "표고 낮을수록 침수위험↑",
    "slope_mean":   "경사 완만할수록 물 정체↑",
    "tpi":          "주변보다 낮을수록(TPI 음수) 위험↑",
    "pump_n_1000m": "펌프 많을수록 대응역량↑ → 취약 관점 반전",
    # pump_dist_m 은 반전하지 않음: '거리 멀다 = 대응 취약 = 취약점수↑' 가 이미 정방향
}
meta = []
Xs = pd.DataFrame(index=df.index)
for c in USE:
    v = df[c].astype(float)
    sk = v.skew()
    use_log = abs(sk) >= 2 and v.min() >= 0
    vv = np.log1p(v) if use_log else v
    lo, hi = np.percentile(vv, [5, 95])
    if hi <= lo:                                   # 분산 거의 없는 피처 방어
        lo, hi = vv.min(), max(vv.max(), vv.min() + 1e-9)
    s = np.clip((vv - lo) / (hi - lo), 0, 1)
    if c in INVERT:
        s = 1 - s
    Xs[c + "_s"] = s
    grp = next(g for g, cols in GROUPS.items() if c in cols)
    meta.append(dict(feature=c, group=grp, skew=round(float(sk), 2), log1p=use_log,
                     p5=round(float(lo), 4), p95=round(float(hi), 4),
                     inverted=c in INVERT, invert_reason=INVERT.get(c, "")))
mt = pd.DataFrame(meta)
P(f"- log1p 적용 {int(mt.log1p.sum())}개 / 부호반전 {int(mt.inverted.sum())}개 (위험↑=점수↑ 통일)")
P("\n| 피처 | 그룹 | 왜도 | log1p | 반전 | 반전사유 |")
P("|---|---|--:|:-:|:-:|---|")
for _, r in mt.iterrows():
    P(f"| {r['feature']} | {r['group']} | {r['skew']} | {'O' if r['log1p'] else ''} "
      f"| {'O' if r['inverted'] else ''} | {r['invert_reason']} |")

out = pd.concat([df, Xs], axis=1)
out.to_parquet(OUT / "features.parquet", index=False)
mt.to_csv(OUT / "feature_meta.csv", index=False, encoding="utf-8-sig")
json.dump({k: v for k, v in GROUPS.items()}, open(OUT / "feature_groups.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

# ---------------------------------------------------------------
# 5. 검증
# ---------------------------------------------------------------
P("\n## 6. 검증")
sc = [c for c in out.columns if c.endswith("_s")]
assert out[sc].min().min() >= -1e-9 and out[sc].max().max() <= 1 + 1e-9, "정규화 범위 이탈"
P(f"- 정규화 피처 {len(sc)}개 전부 [0,1] ✓")
# 방향 검증: 정규화 후 침수O 평균이 침수X보다 커야 '위험↑=점수↑'가 성립
P("\n방향 검증 (침수흔적 격자에서 점수가 더 높아야 정상):")
bad_dir = []
for c in sc:
    a = out.loc[out.trace_flag == 1, c].mean(); z = out.loc[out.trace_flag == 0, c].mean()
    ok = a > z
    if not ok:
        bad_dir.append(c)
    P(f"  {'OK ' if ok else '역전'} {c:<26} 침수O {a:.3f} | 침수X {z:.3f} | {a-z:+.3f}")
P(f"\n- 방향 역전 피처 {len(bad_dir)}개: {bad_dir}")
P("  ※ 역전은 오류가 아니라 '그 지표가 부산에서 침수를 설명하지 못한다'는 신호. M2 중요도에서 재확인.")
P(f"\n- 산출: {OUT/'features.parquet'} ({out.shape}), feature_meta.csv, feature_groups.json")

(REP / "M1_피처준비.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'M1_피처준비.md'}", flush=True)
