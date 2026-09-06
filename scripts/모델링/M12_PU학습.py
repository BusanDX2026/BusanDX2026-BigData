# -*- coding: utf-8 -*-
"""
M12. PU 학습 (Positive-Unlabeled) — 라벨을 '미확인'으로 모델링

■ 문제
  지금 M9 는 침수흔적이 없는 격자를 "안 잠김(음성)"으로 학습한다.
  실제로는 **"기록이 없을 뿐"**이다. 기장군이 기록의 31%인데 인구는 5%인
  조사편향(FIND-3)을 생각하면, 음성 78,331 중 상당수가 '미확인 양성'이다.
  → 표준 이진분류는 이 라벨을 액면 그대로 믿는다. PU 학습은 안 믿는다.

■ 방법 (Elkan & Noto 2008)
  s=1 라벨(관측), y=1 진짜 양성. c = P(s=1|y=1) = 라벨링 확률.
  비라벨 x 가 실제 양성일 확률:  w(x) = ((1-c)/c) · g(x)/(1-g(x))
  비라벨 1건을 (양성, w) + (음성, 1-w) 두 행으로 복제해 가중 학습한다.

  ⚠ 단순 스케일링 f = g/c 는 **단조변환**이라 PR-AUC·ROC·Top-K 가 한 톨도
    안 변한다. 순위를 바꾸려면 반드시 위 가중 재학습을 해야 한다. (A안으로 실증)

  ⚠ SCAR 가정(라벨링이 완전 무작위)은 우리 데이터에서 깨진다 — 자치구마다
    조사 강도가 다르다. 그래서 c 를 **자치구별로** 추정하는 SAR 변형도 같이 본다.

■ 누수 차단
  w(x) 계산에 쓰는 g 는 **외부 폴드 학습셋 안에서만** 내부 GroupKFold 로 구한다.
  전체 OOF 로 w 를 만들면 학습샘플 가중치에 테스트 폴드 정보가 새어든다.
  (MS4 에서 겪은 것과 같은 종류의 누수)

■ 평가 — 라벨이 편향됐으므로 편향된 라벨에 대한 PR-AUC 만 보면 안 된다
  PU 가 제대로 작동하면 **측정 PR-AUC 는 오히려 내려갈 수 있다**(라벨 잡음을
  일부러 안 맞추므로). 그래서 편향에 덜 민감한 보조 지표를 함께 본다:
    · 자치구 내부 Spearman  — 구별 조사강도 차이를 상쇄
    · 재해위험지구 일치도   — 행정의 독립적 판단과의 정합

- 입력: 04_모델/features_v4.parquet
- 출력: 04_모델/M12_PU_oof.parquet, _리포트/M12_PU학습.md
"""
import sys, io, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import spearmanr
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "공공데이터" / "가공데이터"
MOD = GG / "04_모델"; REP = GG / "_리포트"
RS = 42
log = []
def P(s=""):
    print(s, flush=True); log.append(str(s))

P("# M12. PU 학습 (Positive-Unlabeled)")
P(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

df = pd.read_parquet(MOD / "features_v4.parquet")
cfg = json.load(open(MOD / "M3_최종설정.json", encoding="utf-8"))
LC = ["imperv_ratio", "agri_ratio", "paddy_ratio", "forest_ratio", "water_ratio", "road_ratio"]
FEATS = cfg["features"] + [c + "_s" for c in LC] + ["mh_no_dredge_ratio_s"]

X = df[FEATS].values
s = df.trace_flag.values.astype(int)          # 관측 라벨
groups = df.sgg_cd.values
gkf = list(GroupKFold(n_splits=5).split(X, s, groups))
tie = np.random.RandomState(RS).rand(len(s))
SPW = cfg["scale_pos_weight"]
P(f"격자 {len(df):,} · 피처 {len(FEATS)} · 라벨양성 {s.sum():,} ({s.mean():.2%})\n")


def mk(spw=None, **kw):
    p = dict(cfg["params"]); p.update(kw)
    return XGBClassifier(**p, scale_pos_weight=(spw if spw else 1.0),
                         eval_metric="aucpr", tree_method="hist", reg_lambda=1.0,
                         n_jobs=-1, random_state=RS, verbosity=0)


# 재해위험지구 (독립 앵커)
hz = pd.read_parquet(MOD / "features.parquet")[["grid_id", "hazdist_flag"]]
df = df.merge(hz, on="grid_id", how="left", suffixes=("", "_h"))
hzf = (df.get("hazdist_flag", pd.Series(0, index=df.index)).fillna(0).values > 0)
tg = pd.read_parquet(GG / "03_마스터" / "target_grid.parquet")[["grid_id", "trace_area_ratio"]]
df = df.merge(tg, on="grid_id", how="left")
tar = df.trace_area_ratio.fillna(0).values


def report(name, score):
    """편향에 민감한 지표 + 덜 민감한 보조 지표를 함께."""
    sc = score + tie * max(np.ptp(score), 1e-9) * 1e-9
    order = np.argsort(-sc); k = int(len(s) * 0.10)
    top = order[:k]; tp = int(s[top].sum())
    # 자치구 내부 Spearman (조사강도 상쇄)
    rhos = []
    for g in np.unique(groups):
        m = groups == g
        if (tar[m] > 0).sum() >= 15:
            rhos.append(spearmanr(score[m], tar[m]).correlation)
    # 재해위험지구 일치 — 행정 지정 967격자 중 상위10%에 든 비율
    agree = (hzf & np.isin(np.arange(len(s)), top)).sum() / max(hzf.sum(), 1)
    return dict(name=name, pr=average_precision_score(s, score),
                roc=roc_auc_score(s, score), rec10=tp/s.sum(), prec10=tp/k,
                rho=float(np.median(rhos)), agree=agree)


RESULTS = []

# ===============================================================
# 0. 기준 — 현행 M9 (spw=27, 라벨을 액면 그대로 믿음)
# ===============================================================
P("## 0. 기준 M9")
oof_base = np.zeros(len(s))
for tr, te in gkf:
    m = clone(mk(SPW)); m.fit(X[tr], s[tr]); oof_base[te] = m.predict_proba(X[te])[:, 1]
RESULTS.append(report("기준 M9 (spw=27)", oof_base))
P(f"- PR-AUC {RESULTS[-1]['pr']:.4f}\n")

# ===============================================================
# A. 단순 스케일링 f = g/c  — 순위 불변임을 실증
# ===============================================================
P("## A. Elkan-Noto 단순 스케일링 (f = g/c)")
oof_g = np.zeros(len(s))
for tr, te in gkf:
    m = clone(mk(None)); m.fit(X[tr], s[tr]); oof_g[te] = m.predict_proba(X[te])[:, 1]
c_glob = float(oof_g[s == 1].mean())
P(f"- 비가중 g 학습 → c = P(s=1|y=1) 추정 = **{c_glob:.4f}**")
P(f"  (라벨된 양성의 평균 g. 낮을수록 '기록 누락이 많다'는 뜻)")
# ⚠ 여기서 np.clip(g/c, 0, 1) 을 쓰면 안 된다. c=0.223 이라 g>0.223 인 상위 3,181격자
#   (3.9%)가 전부 1.0 에 묶여 **순위가 파괴**된다. PR-AUC 가 0.2999→0.2340 으로 떨어지는데
#   이건 방법의 성질이 아니라 클리핑 버그다. 나눗셈은 양의 상수배라 그대로 두면 순위 불변.
r_scale = report("A. g/c 스케일링", oof_g / max(c_glob, 1e-6))
r_graw = report("(참고) 비가중 g 원본", oof_g)
RESULTS += [r_graw, r_scale]
P(f"- g 원본 PR-AUC {r_graw['pr']:.6f} / g÷c PR-AUC {r_scale['pr']:.6f} "
  f"→ **차이 {abs(r_graw['pr']-r_scale['pr']):.2e}**")
P("  → 양의 상수로 나누는 단조변환이라 순위 지표가 **한 톨도** 안 변한다. 확률만 바뀐다.")
P("    즉 Elkan-Noto 단순 스케일링은 확률 보정용이지 순위 개선 수단이 아니다.\n")

# ===============================================================
# B. 가중 PU (Elkan-Noto 복제) — 누수 차단 중첩 CV
# ===============================================================
def pu_weighted(per_district=False):
    """외부 폴드마다: 학습셋 안에서만 내부 GroupKFold 로 g·c 추정 → 가중 복제 → 재학습."""
    oof = np.zeros(len(s)); cs = []
    for tr, te in gkf:
        Xtr, str_, gtr = X[tr], s[tr], groups[tr]
        # 내부 OOF g (학습셋 한정)
        g_in = np.zeros(len(tr))
        for itr, ite in GroupKFold(n_splits=4).split(Xtr, str_, gtr):
            mi = clone(mk(None)); mi.fit(Xtr[itr], str_[itr])
            g_in[ite] = mi.predict_proba(Xtr[ite])[:, 1]
        # c 추정
        if per_district:
            cmap = {}
            for g in np.unique(gtr):
                mm = (gtr == g) & (str_ == 1)
                cmap[g] = float(g_in[mm].mean()) if mm.sum() >= 20 else float(g_in[str_ == 1].mean())
            cvec = np.array([cmap[g] for g in gtr])
        else:
            cvec = np.full(len(tr), float(g_in[str_ == 1].mean()))
        cs.append(float(np.median(cvec)))
        cvec = np.clip(cvec, 1e-3, 0.999)
        # 비라벨의 양성 확률
        gg = np.clip(g_in, 1e-6, 1 - 1e-6)
        w = ((1 - cvec) / cvec) * (gg / (1 - gg))
        w = np.clip(w, 0, 1)
        unl = str_ == 0
        # 라벨양성 1행 + 비라벨 2행(양성 w / 음성 1-w)
        Xw = np.vstack([Xtr[str_ == 1], Xtr[unl], Xtr[unl]])
        yw = np.concatenate([np.ones(int((str_ == 1).sum())),
                             np.ones(int(unl.sum())), np.zeros(int(unl.sum()))])
        sw = np.concatenate([np.ones(int((str_ == 1).sum())) * SPW,
                             w[unl] * SPW, (1 - w[unl])])
        mf = clone(mk(None)); mf.fit(Xw, yw, sample_weight=sw)
        oof[te] = mf.predict_proba(X[te])[:, 1]
    return oof, float(np.mean(cs))


P("## B. 가중 PU (Elkan-Noto 복제, 중첩 CV로 누수 차단)")
oof_pu, c_b = pu_weighted(False)
RESULTS.append(report("B. 가중 PU (전역 c)", oof_pu))
P(f"- 폴드 평균 c = {c_b:.4f} → PR-AUC {RESULTS[-1]['pr']:.4f}\n")

P("## C. 가중 PU + 자치구별 c (SAR — 조사강도가 구마다 다름)")
oof_pud, c_c = pu_weighted(True)
RESULTS.append(report("C. 가중 PU (자치구별 c)", oof_pud))
P(f"- 폴드 c 중앙값 평균 = {c_c:.4f} → PR-AUC {RESULTS[-1]['pr']:.4f}\n")

# ===============================================================
# D. Bagging PU — 비라벨에서 음성을 반복 표집해 앙상블
# ===============================================================
P("## D. Bagging PU (비라벨 부분표집 앙상블, 20회)")
oof_bag = np.zeros(len(s))
for tr, te in gkf:
    Xtr, str_ = X[tr], s[tr]
    pos = np.where(str_ == 1)[0]; unl = np.where(str_ == 0)[0]
    rng = np.random.RandomState(RS)
    acc = np.zeros(len(te))
    for b in range(20):
        samp = rng.choice(unl, size=len(pos) * 3, replace=False)
        idx = np.concatenate([pos, samp])
        mb = clone(mk(None, n_estimators=200))
        mb.fit(Xtr[idx], str_[idx])
        acc += mb.predict_proba(X[te])[:, 1]
    oof_bag[te] = acc / 20
RESULTS.append(report("D. Bagging PU", oof_bag))
P(f"- PR-AUC {RESULTS[-1]['pr']:.4f}\n")

# ===============================================================
# 결과
# ===============================================================
P("## 결과 종합")
P("| 방법 | PR-AUC | 변화 | ROC-AUC | Top10% 포착 | 정밀도 | 자치구내 ρ(중앙) | 재해위험지구 일치 |")
P("|---|--:|--:|--:|--:|--:|--:|--:|")
b = RESULTS[0]["pr"]
for r in RESULTS:
    ch = "—" if r["name"].startswith("기준") else f"{(r['pr']-b)/b:+.1%}"
    P(f"| {r['name']} | {r['pr']:.4f} | {ch} | {r['roc']:.4f} | {r['rec10']:.1%} | "
      f"{r['prec10']:.1%} | {r['rho']:+.3f} | {r['agree']:.1%} |")

best = max(RESULTS, key=lambda r: r["pr"])
P(f"\n**PR-AUC 최고: {best['name']} ({best['pr']:.4f})**")
bestr = max(RESULTS, key=lambda r: r["rho"])
P(f"**자치구내 ρ 최고: {bestr['name']} ({bestr['rho']:+.3f})** "
  f"— 조사편향에 덜 민감한 지표 기준")

pd.DataFrame({"grid_id": df.grid_id, "oof_base": oof_base, "oof_g": oof_g,
              "oof_pu": oof_pu, "oof_pu_sgg": oof_pud, "oof_bag": oof_bag}
             ).to_parquet(MOD / "M12_PU_oof.parquet", index=False)
REP.mkdir(parents=True, exist_ok=True)
(REP / "M12_PU학습.md").write_text("\n".join(log), encoding="utf-8")
P("\n→ M12_PU_oof.parquet · M12_PU학습.md 저장")
