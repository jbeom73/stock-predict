# scripts/tune_neighbor_beta.py
# -*- coding: utf-8 -*-
"""
Learn softmax temperature β for neighbor weighting in Scene A (forecast).
- For many past windows (queries), we:
  * build the query embedding (EXACT W rows, OHLCV only),
  * retrieve Top-N similar windows,
  * compute neighbors' forward returns (e.g., R10),
  * for each β in grid, form Ř(β) = Σ softmax(β·sim_i)·R_i
  * compare Ř(β) with the query's true forward return R_true
- Pick β that maximizes Pearson correlation between Ř and R_true.
- Saves: models/neighbor_temp_beta.npy
"""

from pathlib import Path
import numpy as np
import pandas as pd
import json, math, random, time
from tqdm import tqdm

from pattern.embed_candle import embed_window_candle
from pattern.indexer import search_faiss
import faiss  # 차원 확인용

# -------- Config (일부는 메타와 동기화됨) --------
ROOT = Path(__file__).resolve().parents[1]
DATA_FEAT = ROOT / "data" / "features"
INDEX_DIR = ROOT / "index"
MODELS    = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

INDEX_NAME = "W60_retstats_candle84"
# WINDOW는 메타에서 강제 동기화함
HORIZON = 10            # R10 기준으로 튜닝 (원하면 5/20으로 바꿔 재학습)
TOPN = 150              # 이웃 검색 개수
SIM_CUTOFF = 0.90       # 너무 느슨하면 노이즈↑, 너무 빡세면 표본↓
MIN_SEP_DAYS = 10       # 동일 종목/근접 창 제거
MAX_QUERIES = 1000      # 전체 메타 중 샘플링 수(속도/안정 타협)

BETA_GRID = [4, 6, 8, 10, 12, 15, 18, 22]  # 그리드 (필요시 조정)
RANDOM_SEED = 42

FEATURES_OHLCV = ["open","high","low","close","volume"]

# -------- Utils --------
def load_feat(ticker: str) -> pd.DataFrame:
    fp = DATA_FEAT / f"{ticker}.parquet"
    df = pd.read_parquet(fp)
    if "date" not in df.columns:
        raise RuntimeError(f"[bad features] {fp} must contain 'date'")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    # 기본 정리
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return df

def slice_window_exact(df: pd.DataFrame, start, W: int) -> pd.DataFrame:
    """start 이상 첫 날부터 정확히 W개 슬라이스 (거래일 결측에도 안전)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()
    s = pd.to_datetime(start, errors="coerce")
    if pd.isna(s):
        return pd.DataFrame()
    s = s.normalize()
    left = df.index.searchsorted(s, side="left")
    left = max(0, min(left, len(df) - W))
    right = left + W
    return df.iloc[left:right]

def window_ohlcv_exact(df: pd.DataFrame, start, W: int) -> pd.DataFrame:
    seg = slice_window_exact(df, start, W)
    if seg.empty or len(seg) < W:
        return pd.DataFrame()
    seg = seg[FEATURES_OHLCV].dropna()
    # 혹시 결측이 섞여 길이가 줄었다면 실패 처리
    if len(seg) != W:
        return pd.DataFrame()
    return seg

def forward_return(close: pd.Series, end_dt, fwd=10):
    idx = close.index
    pos = idx.searchsorted(pd.to_datetime(end_dt), side="right")
    if pos >= len(idx):
        return np.nan
    end = min(pos+fwd, len(idx)-1)
    a = close.iloc[pos:end+1].astype(float).to_numpy()
    if a.size < 2:
        return np.nan
    return float(a[-1]/a[0]-1.0)

def softmax_beta(sims: np.ndarray, beta: float):
    x = beta * sims
    x = x - x.max()  # 안정화
    w = np.exp(x)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w)/len(w)

# -------- Main --------
def main():
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

    # Load meta & enforce WINDOW sync
    meta_path = INDEX_DIR / f"{INDEX_NAME}.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    items = meta["items"]            # list of [ticker, start, end]
    WINDOW = int(meta.get("window", 60))  # 메타 기준 강제
    metric = (meta.get("metric") or "").lower()
    is_cosine = (metric == "cosine") or bool(meta.get("normalized", metric == "cosine"))

    print(f"[meta] windows: {len(items)}  window={WINDOW}  metric={metric or 'unknown'}")

    # Sample query windows for speed
    if MAX_QUERIES and len(items) > MAX_QUERIES:
        idxs = sorted(random.sample(range(len(items)), MAX_QUERIES))
        q_items = [items[i] for i in idxs]
    else:
        q_items = items

    # Preload features by ticker map (speed up IO)
    tickers = sorted({it[0] for it in q_items})
    feat_map = {t: load_feat(t) for t in tickers}

    # --- Sanity: index dimension vs embedding dimension (first query-only) ---
    # 인덱스 차원
    faiss_idx_path = INDEX_DIR / f"{INDEX_NAME}.idx"
    faiss_index = faiss.read_index(str(faiss_idx_path))
    index_dim = faiss_index.d
    del faiss_index

    true_R = []                          # query true forward return
    wret_by_beta = {b: [] for b in BETA_GRID}  # predicted return by beta

    t0 = time.time()
    first_checked = False

    for (tkr, s, e) in tqdm(q_items, desc="queries"):
        df = feat_map[tkr]
        # 쿼리 윈도우: 정확히 W행, OHLCV 5열만
        seg = window_ohlcv_exact(df, s, WINDOW)
        if seg.empty or len(seg) != WINDOW:
            continue

        q_in = np.ascontiguousarray(seg.to_numpy(np.float32))
        q_vec = embed_window_candle(q_in, vol_mode="zlog").reshape(1, -1).astype("float32")

        # 첫 쿼리에서 차원 일치 검사
        if not first_checked:
            d = int(q_vec.shape[1])
            if d != index_dim:
                raise SystemExit(
                    f"[fatal] embedding dim mismatch: query={d}, index={index_dim}\n"
                    f"- 원인: (1) 다른 embed 버전으로 인덱스를 빌드했거나, (2) WINDOW/컬럼/vol_mode 불일치.\n"
                    f"- 조치: build_index.py와 tune_neighbor_beta.py의 embed 설정(OHLCV, vol_mode, WINDOW)을 통일하고 인덱스를 재빌드하세요."
                )
            first_checked = True

        # query true forward return
        R_true = forward_return(df["close"], e, fwd=HORIZON)
        if np.isnan(R_true):
            continue

        # neighbors
        scores, idxs = search_faiss(q_vec, INDEX_NAME, topk=TOPN)
        if scores is None or len(scores) == 0:
            continue

        # collect neighbors with forward returns
        neigh = []
        for sc, ix in zip(np.asarray(scores).reshape(-1), np.asarray(idxs).reshape(-1).astype(int)):
            t2, s2, e2 = items[int(ix)]
            # exclude itself (same ticker & same period) or extremely same
            if (t2 == tkr) and (s2 == s) and (e2 == e):
                continue
            if is_cosine and sc < SIM_CUTOFF:
                continue

            df2 = feat_map.get(t2)
            if df2 is None:
                df2 = load_feat(t2)
                feat_map[t2] = df2

            r_i = forward_return(df2["close"], e2, fwd=HORIZON)
            if np.isnan(r_i):
                continue
            neigh.append((t2, s2, e2, float(sc), float(r_i)))

        if not neigh:
            continue

        sims = np.array([x[3] for x in neigh], dtype=np.float32)
        rets = np.array([x[4] for x in neigh], dtype=np.float32)

        for b in BETA_GRID:
            w = softmax_beta(sims, b)
            R_hat = float(np.sum(w * rets))
            wret_by_beta[b].append(R_hat)

        true_R.append(R_true)

    if not true_R:
        raise SystemExit("[error] no training pairs; check data/index and feature alignment.")

    y = np.array(true_R, dtype=np.float32)
    best_b, best_corr = None, -999.0
    for b in BETA_GRID:
        yhat = np.array(wret_by_beta[b], dtype=np.float32)
        n = min(len(y), len(yhat))
        if n < 20:
            continue
        corr = float(np.corrcoef(y[:n], yhat[:n])[0,1])
        print(f"[eval] beta={b:>4}  corr={corr: .4f}  n={n}")
        if not math.isnan(corr) and corr > best_corr:
            best_corr = corr; best_b = b

    if best_b is None:
        raise SystemExit("[error] no valid beta found")

    out = MODELS / "neighbor_temp_beta.npy"
    np.save(out, np.array([best_b], dtype="float32"))
    print(f"[ok] best β={best_b} (corr={best_corr: .4f}) → saved {out}")
    print(f"[done] elapsed {time.time()-t0: .1f}s")

if __name__ == "__main__":
    # 그냥 Run
    main()
