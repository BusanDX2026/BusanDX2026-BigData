# -*- coding: utf-8 -*-
"""
S8. 마스터 격자 품질 리포트
- 입력: 03_마스터/master_grid.parquet, target_grid.parquet
- 출력: _리포트/S8_품질.md, _리포트/S8_상관행렬.csv
- 내용: 결측·0비율·분포·왜도, Spearman 상관(|r|>0.7 경보), VIF, 타깃과의 단순 상관(참고)
- 규약: S0 §6(정규화·log1p 후보 식별), §9(다중공선성 점검)
"""
import sys, io, warnings
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

GG = Path(__file__).resolve().parents[2] / "공공데이터" / "가공데이터"
OUT = GG / "03_마스터"
REP = GG / "_리포트"
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# S8. 마스터 격자 품질 리포트")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

m = pd.read_parquet(OUT / "master_grid.parquet")
ID = ["grid_id", "sgg_cd", "sgg_nm", "adm_cd", "adm_nm", "x_cen", "y_cen"]
feat = [c for c in m.columns if c not in ID]
P(f"격자 {len(m):,} · 피처 {len(feat)}개\n")

# ---- 1. 기술통계 ----
P("## 1. 피처별 기술통계")
P("| 피처 | 결측% | 0값% | min | median | max | mean | 왜도 | 비고 |")
P("|---|--:|--:|--:|--:|--:|--:|--:|---|")
skew_hi = []
for c in feat:
    s = m[c]
    miss = s.isna().mean() * 100
    v = s.dropna()
    zpct = (v == 0).mean() * 100 if len(v) else np.nan
    sk = v.skew() if len(v) > 2 else np.nan
    note = ""
    if abs(sk) >= 2:
        note = "log1p 권장"; skew_hi.append(c)
    if miss >= 50:
        note = (note + "; " if note else "") + "고결측(부분관측)"
    P(f"| {c} | {miss:.1f} | {zpct:.0f} | {v.min():.3g} | {v.median():.3g} | {v.max():.3g} | {v.mean():.3g} | {sk:.2f} | {note} |")
P(f"\n- log1p 후보(|왜도|≥2): {skew_hi}")

# ---- 2. Spearman 상관 ----
P("\n## 2. 상관행렬 (Spearman) — 다중공선성 점검")
num = m[feat].select_dtypes("number")
corr = num.corr(method="spearman")
corr.round(3).to_csv(REP / "S8_상관행렬.csv", encoding="utf-8-sig")
pairs = []
cols = corr.columns.tolist()
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        r = corr.iloc[i, j]
        if abs(r) >= 0.7:
            pairs.append((cols[i], cols[j], r))
pairs.sort(key=lambda t: -abs(t[2]))
if pairs:
    P("| 피처 A | 피처 B | Spearman r |")
    P("|---|---|--:|")
    for a, b, r in pairs:
        P(f"| {a} | {b} | {r:+.3f} |")
    P("\n→ 위 쌍은 M2/M4에서 개념 그룹핑 or 1개 제거 or VIF 기반 정리. 트리모델은 예측엔 영향 작음.")
else:
    P("|r|≥0.7 쌍 없음.")

# ---- 3. VIF ----
P("\n## 3. VIF (분산팽창) — 고결측 피처 제외, 중앙값 대체 후")
vif_cols = [c for c in num.columns if m[c].isna().mean() < 0.4]
X = num[vif_cols].fillna(num[vif_cols].median())
X = X.loc[:, X.std() > 0]
Xz = (X - X.mean()) / X.std()
Xm = Xz.values
vifs = []
for k in range(Xm.shape[1]):
    y = Xm[:, k]; Z = np.delete(Xm, k, axis=1)
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    r2 = 1 - np.sum((y - Z @ beta) ** 2) / np.sum(y ** 2)
    vifs.append((X.columns[k], 1 / max(1 - r2, 1e-9)))
vifs.sort(key=lambda t: -t[1])
P("| 피처 | VIF |")
P("|---|--:|")
for c, v in vifs:
    P(f"| {c} | {v:.1f} |{' ⚠️' if v >= 10 else ''}")
P("\n→ VIF≥10 은 M4 피처정리 대상 (연면적/인구/인구밀도 등 파생 중복 예상).")

# ---- 4. 타깃과의 단순 상관 (참고) ----
P("\n## 4. 침수흔적(target)과의 단순 상관 — 참고용 (정식 평가는 모델링 M6)")
t = pd.read_parquet(OUT / "target_grid.parquet")[["grid_id", "trace_flag"]]
mt = m.merge(t, on="grid_id", how="left")
mt["trace_flag"] = mt["trace_flag"].fillna(0)
rows = []
for c in num.columns:
    r = mt[[c, "trace_flag"]].dropna().corr(method="spearman").iloc[0, 1]
    rows.append((c, r))
rows.sort(key=lambda t: -abs(t[1]))
P("| 피처 | Spearman r (vs trace_flag) |")
P("|---|--:|")
for c, r in rows:
    P(f"| {c} | {r:+.3f} |")
P("\n→ 부호·크기가 상식과 맞는지 확인용. 공간 자기상관 때문에 이 값 자체를 성능지표로 쓰지 않음(M6 블록 홀드아웃).")

# ---- 5. 규약 체크 ----
P("\n## 5. S0 §9 체크")
P(f"- CRS/격자 수 보존: master {len(m):,} (S4 격자 81,435)")
P(f"- target leakage: master에 trace_*/danger_level 없음 = {not any(k in feat for k in ['trace_flag','trace_area_ratio','trace_max_depth','danger_level'])}")
P(f"- 고결측 피처(부분관측, 모델 단계 처리): {[c for c in feat if m[c].isna().mean()>=0.5]}")
P(f"- 상관행렬: {REP/'S8_상관행렬.csv'}")

(REP / "S8_품질.md").write_text("\n".join(log), encoding="utf-8")
print(f"\n==> 리포트: {REP/'S8_품질.md'}", flush=True)
