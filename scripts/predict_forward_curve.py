# scripts/predict_forward_curve.py
# -*- coding: utf-8 -*-
"""
Forward curve prediction (hybrid-ranked neighbors) with robust fallback.
- 최근 W일 타깃 과거: 실선
- 예측선(softmax 가중 기대값): 점선, t=0 연속
- 분포 밴드: P10–P90 / P25–P75
- 하이브리드 정렬: sim01 + (최적 래그 상관) - 래그 벌점
- 후보 없음 상황을 방지하기 위해 단계적 완화(backoff) 로직 포함
"""

from __future__ import annotations
from pathlib import Path
import json, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pattern.embed_candle import embed_window_candle
from pattern.indexer import search_faiss

# ---------------- Config ----------------
INDEX_NAME = "W60_retstats_candle84"
TARGET = "005930"
WINDOW = 60
MAX_FWD = 20

# 1차 검색/이웃 수
TOPN_SEARCH = 800             # 1차 검색량(여유있게)
TOPN_NEIGH  = 120             # 최종 이웃 수(하이브리드 정렬 후)

# 컷오프/중복 방지
SIM_CUTOFF = 0.90             # 코사인에서만 적용
SELF_NEAR_DAYS = 8            # 타깃 종료일과 근접 자기자신 제거
EXCLUDE_RECENT_DAYS = 0       # 타깃 최근 W일 사용 권장(=0)

# 하이브리드 스코어
LAG_MAX = 5
W_SIM = 0.4
W_CORR = 0.6
LAG_PEN = 0.10

# softmax β (학습값이 있으면 사용)
ROOT = Path(__file__).resolve().parents[1]
BETA_PATH = ROOT / "models" / "neighbor_temp_beta.npy"
DEFAULT_BETA = 10.0

# 경로
DATA_FEAT = ROOT / "data" / "features"
INDEX_DIR = ROOT / "index"
OUT_DIR = ROOT / "outputs"; OUT_DIR.mkdir(exist_ok=True)

FEATURES_OHLCV = ["open","high","low","close","volume"]

# ---------------- Utils ----------------
def _setup_korean_font():
    try:
        from matplotlib import rcParams
        sys = platform.system()
        rcParams["font.family"] = "Malgun Gothic" if sys=="Windows" else ("AppleGothic" if sys=="Darwin" else rcParams["font.family"])
        rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

def _read_feature_parquet(ticker: str) -> pd.DataFrame:
    fp = DATA_FEAT / f"{ticker}.parquet"
    if not fp.exists(): raise FileNotFoundError(f"[features missing] {fp}")
    df = pd.read_parquet(fp)
    if not isinstance(df.index, pd.DatetimeIndex):
        col = next((c for c in ["date","Date","datetime","Datetime","time","Time"] if c in df.columns), None)
        if col is None:
            for c in df.columns:
                if np.issubdtype(df[c].dtype, np.datetime64): col = c; break
        if col is None: raise ValueError(f"[bad index] {fp}: DatetimeIndex/날짜 컬럼 필요")
        df[col] = pd.to_datetime(df[col], errors="coerce"); df = df.set_index(col)
    return df.sort_index().replace([np.inf,-np.inf], np.nan).ffill().bfill()

def load_feat(ticker: str) -> pd.DataFrame:
    return _read_feature_parquet(ticker)

def get_target_window_matrix(df: pd.DataFrame, W: int, ex_recent_days: int):
    df2 = df[FEATURES_OHLCV].dropna()
    seg = df2.iloc[-(W+ex_recent_days):-ex_recent_days] if ex_recent_days>0 else df2.iloc[-W:]
    if len(seg) < W: seg = df2.iloc[-W:]
    if len(seg) != W: raise ValueError(f"need {W}, have {len(seg)}")
    return seg.to_numpy(np.float32), seg.index[0], seg.index[-1]

def _load_meta_doc(meta_path: Path):
    m = json.loads(meta_path.read_text(encoding="utf-8"))
    metric = (m.get("metric") or "").lower()
    items = m["items"]
    return {"items": items, "metric": metric, "normalized": bool(m.get("normalized", metric=="cosine"))}

def slice_window_exact(df: pd.DataFrame, start, W: int) -> pd.DataFrame:
    s = pd.to_datetime(start, errors="coerce")
    if pd.isna(s): return pd.DataFrame()
    left = df.index.searchsorted(s.normalize(), side="left")
    left = max(0, min(left, len(df)-W))
    return df.iloc[left:left+W]

def pivot_normalize(win_close: np.ndarray, fwd_close: np.ndarray | None = None):
    base = win_close[-1] if np.isfinite(win_close[-1]) and win_close[-1]!=0 else 1.0
    y_past = (np.asarray(win_close,float)/base - 1.0)*100.0
    y_past -= y_past[-1]
    W = len(win_close); x_past = np.arange(-W+1, 1, dtype=int)
    if fwd_close is None or len(fwd_close)==0: return x_past, y_past, None, None
    y_fwd = (np.asarray(fwd_close,float)/base - 1.0)*100.0
    x_fwd = np.arange(1, len(fwd_close)+1, dtype=int)
    return x_past, y_past, x_fwd, y_fwd

# ---- hybrid helpers ----
def _to_similarity(score: float, is_cosine: bool) -> float:
    if is_cosine: return float(score)
    d = max(0.0, float(score))
    return 1.0/(1.0+d)

def _best_lag_corr(y_ref: np.ndarray, y_cmp: np.ndarray, lag_max: int, min_overlap: int = 20):
    W = min(len(y_ref), len(y_cmp))
    a = np.asarray(y_ref, float)[-W:]; b = np.asarray(y_cmp, float)[-W:]
    best_c, best_l = -2.0, 0
    for l in range(-lag_max, lag_max+1):
        if l < 0:
            ar = a[-l:]; br = b[:len(a)-(-l)]
        elif l > 0:
            ar = a[:len(a)-l]; br = b[l:len(a)]
        else:
            ar, br = a, b
        if len(ar) < max(min_overlap, W//3):
            continue
        c = float(np.corrcoef(ar, br)[0,1])
        if np.isfinite(c) and c > best_c:
            best_c, best_l = c, l
    if not np.isfinite(best_c): best_c, best_l = -1.0, 0
    return best_c, best_l

def _hybrid(sim01: float, corr: float, lag: int, lag_max: int, w_sim: float, w_corr: float, lag_pen: float):
    return w_sim*sim01 + w_corr*((corr+1.0)/2.0) - (abs(lag)/max(1,lag_max))*lag_pen

def _softmax_beta(x: np.ndarray, beta: float):
    z = beta*np.asarray(x, float)
    z -= np.max(z)
    w = np.exp(z); s = np.sum(w)
    return (w/s) if s>0 else np.ones_like(w)/len(w)

# ---- candidate collection with backoff ----
def _collect_candidates_with_backoff(q_vec, index_name, items, is_cosine, t_edt,
                                     topn_search, sim_cutoff, self_near_days):
    """
    단계적 완화:
      1) 기본 설정
      2) cutoff 완화
      3) 검색량 확대
      4) 자기자신 근접 완화
    """
    stages = [
        {"TOPN_SEARCH": topn_search, "SIM_CUTOFF": sim_cutoff, "SELF_NEAR": self_near_days},
        {"TOPN_SEARCH": max(topn_search, 1200), "SIM_CUTOFF": 0.85 if is_cosine else sim_cutoff, "SELF_NEAR": self_near_days},
        {"TOPN_SEARCH": 2000, "SIM_CUTOFF": 0.80 if is_cosine else sim_cutoff, "SELF_NEAR": max(3, self_near_days//2)},
        {"TOPN_SEARCH": 3000, "SIM_CUTOFF": 0.0 if is_cosine else sim_cutoff, "SELF_NEAR": 0},
    ]

    for i, st in enumerate(stages, 1):
        scores, idxs = search_faiss(q_vec, index_name, topk=st["TOPN_SEARCH"])
        scores = np.asarray(scores).reshape(-1); idxs = np.asarray(idxs).astype(int)
        raw=[]
        for sc, ix in zip(scores, idxs):
            rec = items[int(ix)]
            tkr, s, e = (rec if isinstance(rec, (list,tuple)) else (rec["ticker"], rec["start"], rec["end"]))
            # 자기자신/근접자기자신 제거
            if tkr == TARGET:
                if (is_cosine and sc>=0.999999) or ((not is_cosine) and sc<=1e-9):
                    continue
                if st["SELF_NEAR"]>0 and abs((pd.to_datetime(e)-t_edt).days) < st["SELF_NEAR"]:
                    continue
            # 코사인 컷오프
            if is_cosine and sc < st["SIM_CUTOFF"]:
                continue
            raw.append((tkr, s, e, float(sc)))

        print(f"[stage {i}] cand={len(raw)}  topk={st['TOPN_SEARCH']}  cutoff={st['SIM_CUTOFF']}  self_near={st['SELF_NEAR']}")
        if len(raw) >= max(20, TOPN_NEIGH//2):
            return raw  # 충분하면 반환

    return raw  # 마지막 시도 결과(없으면 빈 리스트)

# ---------------- Runner ----------------
def main():
    _setup_korean_font()

    # 타깃 윈도우 & 임베딩
    dft = load_feat(TARGET)
    Wmat, _, t_edt = get_target_window_matrix(dft, WINDOW, EXCLUDE_RECENT_DAYS)
    q_vec = embed_window_candle(Wmat, vol_mode="zlog").reshape(1,-1)

    # 메타/유사도 타입
    meta = _load_meta_doc(INDEX_DIR / f"{INDEX_NAME}.meta.json")
    items = meta["items"]; is_cosine = (meta["metric"]=="cosine") or meta["normalized"]

    # 타깃 과거 %시계열
    seg_t = dft.iloc[-WINDOW:] if EXCLUDE_RECENT_DAYS==0 else dft.iloc[-(WINDOW+EXCLUDE_RECENT_DAYS):-EXCLUDE_RECENT_DAYS]
    t_close = seg_t["close"].to_numpy(float)
    xt_p, yt_p, _, _ = pivot_normalize(t_close, None)

    # 후보 수집(백오프 포함)
    raw = _collect_candidates_with_backoff(
        q_vec, INDEX_NAME, items, is_cosine, t_edt,
        topn_search=TOPN_SEARCH, sim_cutoff=SIM_CUTOFF, self_near_days=SELF_NEAR_DAYS
    )
    if not raw:
        raise SystemExit("[error] no candidates from index (after backoff)")

    # 하이브리드 정렬
    enriched=[]
    for (mt, ms, me, sc) in raw:
        dfm = load_feat(mt)
        seg = slice_window_exact(dfm, ms, WINDOW)
        if seg.empty or len(seg) < WINDOW: continue
        m_close = seg["close"].astype(float).to_numpy()
        _, y_cmp, _, _ = pivot_normalize(m_close[:WINDOW], None)
        corr, lag = _best_lag_corr(yt_p, y_cmp, LAG_MAX)
        sim01 = _to_similarity(sc, is_cosine)
        hyb = _hybrid(sim01, corr, lag, LAG_MAX, W_SIM, W_CORR, LAG_PEN)
        enriched.append((mt, ms, me, sc, sim01, corr, lag, hyb))

    if not enriched:
        raise SystemExit("[error] all candidates invalid during enrichment")

    enriched.sort(key=lambda x: x[7], reverse=True)
    neigh = enriched[:TOPN_NEIGH]

    # β 로드
    beta = DEFAULT_BETA
    if BETA_PATH.exists():
        try:
            v = np.load(BETA_PATH).astype(float).reshape(-1)
            if v.size >= 1 and np.isfinite(v[0]) and v[0] > 0: beta = float(v[0])
        except Exception:
            pass

    # 이웃 미래경로 수집
    fwds, sims01 = [], []
    for (mt, ms, me, sc, sim01, corr, lag, hyb) in neigh:
        dfm = load_feat(mt)
        seg = slice_window_exact(dfm, ms, WINDOW)
        if seg.empty or len(seg)<WINDOW: continue
        pos_end = dfm.index.get_loc(seg.index[-1])
        fwd_close = dfm["close"].astype(float).to_numpy()[pos_end+1 : pos_end+1+MAX_FWD]
        _, _, xf, yf = pivot_normalize(seg["close"].astype(float).to_numpy(), fwd_close)
        if xf is None or yf is None or len(yf)==0:
            continue
        fwds.append(np.asarray(yf, float))
        sims01.append(float(sim01))

    if not fwds:
        raise SystemExit("[error] no neighbor forward paths")

    # 행렬화
    arr = np.full((len(fwds), MAX_FWD), np.nan, dtype=float)
    for i, y in enumerate(fwds):
        L = min(len(y), MAX_FWD); arr[i, :L] = y[:L]

    # 가중 기대값
    w = _softmax_beta(np.asarray(sims01, float), beta)
    y_pred = np.array([
        (np.nansum(arr[:, t] * w) / np.nansum(w[~np.isnan(arr[:, t])])) if np.any(~np.isnan(arr[:, t])) else np.nan
        for t in range(MAX_FWD)
    ], dtype=float)

    # 분포 통계
    p10 = np.nanpercentile(arr, 10, axis=0)
    p25 = np.nanpercentile(arr, 25, axis=0)
    med = np.nanpercentile(arr, 50, axis=0)
    p75 = np.nanpercentile(arr, 75, axis=0)
    p90 = np.nanpercentile(arr, 90, axis=0)
    mean_u = np.nanmean(arr, axis=0)

    # ---------------- Plot ----------------
    plt.figure(figsize=(11.2, 5.8))
    ax = plt.gca(); ax.grid(True, alpha=0.3); ax.margins(x=0.02, y=0.08)

    # 과거: 실선
    ax.plot(xt_p, yt_p, lw=2.2, label=f"{TARGET} (past)")

    # 예측선: 점선, t=0 연속
    xf = np.arange(1, MAX_FWD+1, dtype=int)
    xf_plot = np.concatenate(([0], xf))
    yp_plot = np.concatenate(([0.0], y_pred))
    ax.plot(xf_plot, yp_plot, ls=":", lw=2.2, label=f"Pred (β={beta:g})")

    # 밴드
    ax.fill_between(xf, p10, p90, alpha=0.15, label="P10–P90")
    ax.fill_between(xf, p25, p75, alpha=0.25, label="P25–P75")
    ax.plot(xf, med, lw=1.2, alpha=0.9, label="Median")

    ax.axvline(0, color="k", ls=":", alpha=0.6)
    ax.set_xlabel("Days from pivot (0)"); ax.set_ylabel("Return from pivot (%)")
    ax.set_title(f"Forward Curve · {TARGET} · W={WINDOW} · Neigh={len(sims01)}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    out_png = OUT_DIR / f"forecast_{TARGET}_W{WINDOW}_H{MAX_FWD}.png"
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"[ok] saved plot → {out_png}")

    # ---------------- CSV Save ----------------
    out_csv = OUT_DIR / f"forecast_{TARGET}_W{WINDOW}_H{MAX_FWD}.csv"
    df_stats = pd.DataFrame({
        "t": np.arange(1, MAX_FWD+1, dtype=int),
        "mean_weighted": y_pred,
        "mean_unweighted": mean_u,
        "median": med,
        "p10": p10,
        "p25": p25,
        "p75": p75,
        "p90": p90,
    })
    df_stats.to_csv(out_csv, index=False)
    print(f"[ok] saved stats → {out_csv}")

if __name__ == "__main__":
    main()
