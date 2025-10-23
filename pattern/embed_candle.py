# pattern/embed_candle.py
# -*- coding: utf-8 -*-
"""
==============================================================================
Module: embed_candle.py
Purpose:
    - Create numerical embeddings (feature vectors) that represent
      60-day stock chart patterns, including both price movement,
      candlestick morphology, and normalized volume dynamics.
    - Used by the pattern similarity engine (FAISS) to find visually
      and behaviorally similar chart segments across all tickers.

Embedding Structure (Total 84 dimensions):
    1️⃣ Base (64 dims)
        - Normalized log returns (≈59)
        - 5 summary stats: volatility, slope, p25, p50, p75
    2️⃣ Candle Morphology (16 dims)
        - Body%, shadow ratios, and pattern frequencies
        - hammer, shooting-star, doji, engulfing, gap-up/down, etc.
    3️⃣ Volume Dynamics (4 dims)
        - Z-score(log(volume)) → scale-invariant
        - mean_logv, std_logv, spike_rate, corr(|body|, logv)

Normalization Logic:
    - Price-based features: z-scored log returns
    - Volume-based features: z-scored log(volume)
      → neutralizes scale difference between large-cap/small-cap stocks

Usage:
    from pattern.embed_candle import embed_window_candle
    vec = embed_window_candle(window, vol_mode="zlog")

Args:
    window : np.ndarray of shape (W, F)
        [open, high, low, close, volume, ...] for 60 days
    vol_mode : str
        "zlog"          : log(volume) normalized (default)
        "turnover_zlog" : log(price×volume) normalized
        "ratio"         : volume / mean(volume)

Returns:
    np.ndarray, shape (84,)
        concatenated embedding vector [base64 | candle16 | volume4]
==============================================================================
"""

import numpy as np

EPS = 1e-8


# ---------- Base price-return features ----------
def _base_returns_stats(close):
    """64 dims = 59 (zscored log returns) + 5 summary stats"""
    r = np.diff(np.log(close + EPS))
    r = (r - r.mean()) / (r.std() + EPS)
    vol = r.std()
    slope = np.polyfit(np.arange(len(r)), r, 1)[0]
    p25, p50, p75 = np.percentile(r, [25, 50, 75])
    return np.concatenate([r.astype("float32"),
                           np.array([vol, slope, p25, p50, p75], dtype="float32")])


# ---------- Candle shape / morphology ----------
def _candle_window_stats(open_, high, low, close):
    """Compute 16D summary of candle morphology."""
    W = len(close)
    body = close - open_
    rng = np.maximum(high - low, EPS)
    upper = high - np.maximum(open_, close)
    lower = np.minimum(open_, close) - low

    body_pct = body / (np.maximum(np.abs(open_), EPS))
    upper_ratio = upper / rng
    lower_ratio = lower / rng
    dir_ = np.sign(body)

    # 1) Continuous stats (6)
    c = [
        np.mean(body_pct), np.std(body_pct),
        np.mean(upper_ratio), np.std(upper_ratio),
        np.mean(lower_ratio), np.std(lower_ratio),
    ]

    # 2) Frequency features (10)
    bullish_rate = np.mean(dir_ > 0)
    doji_rate = np.mean(np.abs(body_pct) < 0.1)
    long_upper = np.mean(upper_ratio > 0.6)
    long_lower = np.mean(lower_ratio > 0.6)
    marubozu = np.mean((np.abs(body_pct) > 0.02) &
                       (upper_ratio < 0.1) & (lower_ratio < 0.1))
    hammer_rate = np.mean((lower_ratio > 0.6) & (dir_ > 0))
    star_rate = np.mean((upper_ratio > 0.6) & (dir_ < 0))

    gap_up = gap_dn = engulf_bull = 0.0
    if W >= 2:
        prev_close = close[:-1]
        today_open = open_[1:]
        gap_up = np.mean((today_open / (prev_close + EPS)) > 1.005)
        gap_dn = np.mean((today_open / (prev_close + EPS)) < 0.995)
        prev_body = close[:-1] - open_[:-1]
        today_body = close[1:] - open_[1:]
        engulf_bull = np.mean(
            (prev_body < 0) & (today_body > 0) &
            (np.abs(today_body) > np.abs(prev_body))
        )

    return np.array([
        *c,                      # 6
        bullish_rate, doji_rate, long_upper, long_lower,  # 4 → 10
        marubozu, hammer_rate, star_rate,                 # 3 → 13
        gap_up, gap_dn, engulf_bull                       # 3 → 16
    ], dtype="float32")


# ---------- Volume pattern ----------
def _volume_window_stats(open_, close, volume, price=None, mode="zlog"):
    """
    4D volume summary (scale-invariant)
      [mean_log, std_log, spike_rate, corr(|body|, metric)]
    """
    body_pct = (close - open_) / (np.maximum(np.abs(open_), EPS))

    if mode == "turnover_zlog":
        metric = np.log1p((price if price is not None else close) * volume + EPS)
    elif mode == "ratio":
        metric = volume / (np.mean(volume) + EPS)
        metric = np.log1p(metric)
    else:  # default: "zlog"
        metric = np.log1p(volume + EPS)

    mu = float(np.mean(metric))
    sd = float(np.std(metric))
    z = (metric - mu) / (sd + EPS)
    spike_rate = float(np.mean(z > 2.0))

    if np.std(np.abs(body_pct)) < EPS or sd < EPS:
        corr = 0.0
    else:
        corr = float(np.corrcoef(np.abs(body_pct), metric)[0, 1])

    return np.array([mu, sd, spike_rate, corr], dtype="float32")


# ---------- Main embedding function ----------
def embed_window_candle(window: np.ndarray, vol_mode: str = "zlog"):
    """
    window: (W, F)
        front columns must be [open, high, low, close, volume]
    vol_mode: "zlog" (default), "turnover_zlog", "ratio"
    return: (84,) embedding vector
    """
    open_, high, low, close = [window[:, i].astype("float32") for i in range(4)]
    volume = window[:, 4].astype("float32") if window.shape[1] > 4 else np.zeros_like(close)

    base64 = _base_returns_stats(close)
    can16 = _candle_window_stats(open_, high, low, close)
    vol4 = _volume_window_stats(open_, close, volume, price=close, mode=vol_mode)

    return np.concatenate([base64, can16, vol4]).astype("float32")
