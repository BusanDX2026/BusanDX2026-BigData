# -*- coding: utf-8 -*-
"""
DEM 수문 파생 공용 모듈 (M3 · MS1 공유)

■ 왜 만들었나 (코드리뷰 2026-09-04, 지적 #1·#2)
  기존 M3/MS1에 복제돼 있던 D8 코드는 **하강 조건이 없었다**:
      best_drop = -inf ;  upd = (drop > best_drop) & isfinite & valid
  → drop<0(오르막) 이웃도 채택. 실측 결과 부산 DEM 872,873셀 중 47,852셀(5.5%)이
    하강 이웃 없이 오르막/평탄으로 배수됐고, 로그는 싱크를 42개로 잘못 보고했다.
    그 위에서 계산된 flow_acc·TWI·수계망·HAND가 전부 오염.
  또 HAND는 하류 미해결 시 자기표고로 폴백해 hand=0(최대취약)이 됐다 → 31.5% 격자가 0m.

■ 수정
  1) 우선순위 함몰메움(priority-flood)으로 DEM을 먼저 처리 → 모든 셀에 하강 경로 보장
  2) D8에 `drop > 0` 강제. 남는 무하류 셀은 진짜 유출구(경계·해안)뿐
  3) HAND 전파는 메움표고 오름차순(하류 선처리 보장). 미해결은 **NaN**(0 아님)

함수
  fill_depressions(dem, valid)          -> filled
  d8_directions(z, valid, px)           -> best_nb (평탄/무하류는 -1)
  flow_accumulate(best_nb, z, valid)    -> flow_acc (셀 수)
  hand_from_stream(dem, z, best_nb, stream) -> hand (m, 미해결 NaN)
"""
import heapq
import numpy as np

NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
EPS = 1e-3   # 메움 시 부여하는 최소 경사 (mm 단위, 물리적으로 무시 가능)


def _shift(arr, dr, dc, fill):
    """arr을 (dr,dc)만큼 당겨온 배열 (경계는 fill)."""
    H, W = arr.shape
    out = np.full((H, W), fill, dtype=arr.dtype)
    r0, r1 = max(0, dr), H + min(0, dr)
    c0, c1 = max(0, dc), W + min(0, dc)
    out[r0:r1, c0:c1] = arr[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return out


def fill_depressions(dem, valid):
    """우선순위 함몰메움(Planchon-Darboux/priority-flood).
    경계 및 nodata 인접 유효셀을 유출구로 삼아 낮은 것부터 확장하며
    filled[nb] = max(dem[nb], z_pop + EPS) 를 부여한다.
    반환: filled (무효셀은 NaN)"""
    H, W = dem.shape
    filled = np.full((H, W), np.nan, dtype="float64")
    seen = np.zeros((H, W), dtype=bool)

    # 유출구 시드: 래스터 경계에 있거나, nodata에 인접한 유효셀
    border = np.zeros((H, W), dtype=bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    adj_nodata = np.zeros((H, W), dtype=bool)
    for dr, dc in NB8:
        adj_nodata |= ~_shift(valid, dr, dc, False)
    seed = valid & (border | adj_nodata)

    heap = []
    rs, cs = np.nonzero(seed)
    for r, c in zip(rs.tolist(), cs.tolist()):
        z = float(dem[r, c])
        filled[r, c] = z
        seen[r, c] = True
        heap.append((z, r, c))
    heapq.heapify(heap)

    while heap:
        z, r, c = heapq.heappop(heap)
        for dr, dc in NB8:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                continue
            if seen[nr, nc] or not valid[nr, nc]:
                continue
            nz = dem[nr, nc]
            nz = nz if nz > z + EPS else z + EPS      # 함몰이면 살짝 올려 메움
            filled[nr, nc] = nz
            seen[nr, nc] = True
            heapq.heappush(heap, (float(nz), nr, nc))
    return filled


def d8_directions(z, valid, px):
    """최급경사 하류 이웃 (평평/무하류는 -1). **drop > 0 강제.**
    z는 메움표고를 넣어야 무하류 셀이 유출구만 남는다."""
    H, W = z.shape
    idx = np.arange(H * W).reshape(H, W)
    zz = np.where(valid, z, np.inf)
    best_drop = np.zeros((H, W))          # 0 초기화 = drop>0 인 것만 채택
    best_nb = np.full((H, W), -1, dtype=np.int64)
    for dr, dc in NB8:
        zs = _shift(zz, dr, dc, np.inf)
        ns = _shift(idx, dr, dc, -1)
        dist = np.hypot(dr, dc) * px
        drop = (zz - zs) / dist
        upd = (drop > best_drop) & np.isfinite(zs) & valid
        best_drop = np.where(upd, drop, best_drop)
        best_nb = np.where(upd, ns, best_nb)
    return best_nb


def flow_accumulate(best_nb, z, valid):
    """표고 내림차순 1-pass 누적. 반환: flow_acc(셀 수), 무효셀 NaN."""
    H, W = z.shape
    flat_z = np.where(valid, z, -np.inf).ravel()
    order = np.argsort(-flat_z, kind="stable")
    order = order[np.isfinite(flat_z[order])]
    acc = np.ones(H * W)
    nb = best_nb.ravel()
    for i in order:
        j = nb[i]
        if j >= 0:
            acc[j] += acc[i]
    out = acc.reshape(H, W)
    return np.where(valid, out, np.nan)


def hand_from_stream(dem, z, best_nb, stream, valid):
    """HAND = 원표고 − (흐름경로를 따라 도달하는 수계셀의 원표고).
    z(메움표고) 오름차순으로 처리하면 하류가 항상 먼저 확정된다.
    수계에 도달 못 하는 셀은 **NaN**(호출부에서 대체) — 0으로 채우지 않는다."""
    H, W = dem.shape
    e_orig = np.where(valid, dem, np.nan).ravel()
    e_fill = np.where(valid, z, np.nan).ravel()
    nb = best_nb.ravel()
    st = stream.ravel()

    drain = np.full(H * W, np.nan)
    asc = np.argsort(e_fill, kind="stable")
    asc = asc[np.isfinite(e_fill[asc])]
    for i in asc:
        if st[i]:
            drain[i] = e_orig[i]
        else:
            j = nb[i]
            if j >= 0:
                drain[i] = drain[j]        # 미해결이면 NaN 이 그대로 전파됨
    hand = e_orig - drain
    hand = np.where(np.isfinite(hand), np.clip(hand, 0, None), np.nan)
    return hand.reshape(H, W)


def derive_all(dem, valid, px, stream_km2=1.0, verbose_fn=None):
    """메움 → D8 → 누적 → 수계 → HAND/TWI/수계거리 일괄 산출."""
    from scipy.ndimage import distance_transform_edt
    P = verbose_fn or (lambda *_: None)

    zf = fill_depressions(dem, valid)
    n_filled = int((valid & (zf > dem + EPS)).sum())
    P(f"- 함몰메움: {n_filled:,}셀 상향 ({n_filled/valid.sum():.1%})")

    best_nb = d8_directions(zf, valid, px)
    n_sink = int(((best_nb < 0) & valid).sum())
    P(f"- D8(drop>0 강제): 무하류 셀 {n_sink:,} ({n_sink/valid.sum():.2%}) = 경계·해안 유출구")

    facc = flow_accumulate(best_nb, zf, valid)
    P(f"- 흐름누적 max {np.nanmax(facc):,.0f}셀 (약 {np.nanmax(facc)*px*px/1e6:.0f} km2)")

    thr = stream_km2 * 1e6 / (px * px)
    stream = (facc >= thr) & valid
    P(f"- 수계망(≥{stream_km2} km²): {int(stream.sum()):,}픽셀 ({stream.sum()/valid.sum():.2%})")

    hand = hand_from_stream(dem, zf, best_nb, stream, valid)
    n_nan = int((valid & ~np.isfinite(hand)).sum())
    P(f"- HAND 산출: 미해결 {n_nan:,}셀 ({n_nan/valid.sum():.2%}) → NaN 유지")

    gy, gx = np.gradient(np.where(valid, dem, np.nan), px, px)
    tan_b = np.maximum(np.tan(np.arctan(np.sqrt(gx ** 2 + gy ** 2))), 0.001)
    twi = np.log((facc * px) / tan_b)

    d2s = distance_transform_edt(~stream, sampling=px)
    d2s = np.where(valid, d2s, np.nan)
    return dict(filled=zf, best_nb=best_nb, flow_acc=facc, stream=stream,
                hand=hand, twi=twi, dist_stream=d2s)
