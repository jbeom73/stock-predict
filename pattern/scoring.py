# pattern/scoring.py
# -*- coding: utf-8 -*-
"""
Reusable scoring utilities for pattern-ai.
- best_lag_corr: 피벗정규화 과거구간의 최적 래그 상관
- to_similarity: 거리→유사도 변환
- hybrid_score: (유사도, 상관, 래그벌점) 하이브리드 점수
- rank_with_hybrid: 후보 창 리스트에 하이브리드 점수 부여 및 정렬
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, Tuple, List

def best_lag_corr(y_ref: np.ndarray,
                  y_cmp: np.ndarray,
                  lag_max: int = 5,
                  min_overlap: int = 20) -> Tuple[float, int]:
    """
    y_ref, y_cmp: 길이 W의 %시계열(피벗정규화 과거)
    반환: (최대 피어슨 상관, 해당 래그) ; lag<0는 비교시계열을 '좌로'(과거로) 이동.
    """
    y_ref = np.asarray(y_ref, float)
    y_cmp = np.asarray(y_cmp, float)
    W = min(len(y_ref), len(y_cmp))
    a = y_ref[-W:]; b = y_cmp[-W:]
    best_c, best_l = -2.0, 0
    for l in range(-lag_max, lag_max + 1):
        if l < 0:
            ar = a[-l:]; br = b[:len(a) - (-l)]
        elif l > 0:
            ar = a[:len(a) - l]; br = b[l:len(a)]
        else:
            ar, br = a, b
        if len(ar) < max(min_overlap, W // 3):
            continue
        c = float(np.corrcoef(ar, br)[0, 1])
        if np.isfinite(c) and c > best_c:
            best_c, best_l = c, l
    if not np.isfinite(best_c):
        best_c, best_l = -1.0, 0
    return best_c, best_l

def to_similarity(score: float, is_cosine: bool) -> float:
    """
    검색 스코어를 [0,1] 유사도로 변환.
    - cosine: 그대로 반환(클수록 유사).
    - L2 distance: 1/(1+d)로 간단 정규화.
    """
    if is_cosine:
        return float(score)
    d = max(0.0, float(score))
    return 1.0 / (1.0 + d)

def hybrid_score(sim01: float,
                 corr: float,
                 lag: int,
                 lag_max: int = 5,
                 w_sim: float = 0.4,
                 w_corr: float = 0.6,
                 lag_pen: float = 0.10) -> float:
    """
    하이브리드 점수(클수록 좋음):
      w_sim * sim01 + w_corr * ((corr+1)/2) - (|lag|/lag_max)*lag_pen
    """
    sim_term = w_sim * float(sim01)
    corr_term = w_corr * ((float(corr) + 1.0) / 2.0)  # [-1,1] -> [0,1]
    pen = (abs(int(lag)) / max(1, int(lag_max))) * float(lag_pen)
    return sim_term + corr_term - pen

def rank_with_hybrid(matches: Iterable[tuple],
                     y_ref: np.ndarray,
                     loader,
                     W: int,
                     is_cosine: bool,
                     lag_max: int = 5,
                     w_sim: float = 0.4,
                     w_corr: float = 0.6,
                     lag_pen: float = 0.10,
                     topk: int | None = None) -> List[tuple]:
    """
    matches: [(ticker, start, end, raw_score), ...]
    loader: ticker -> DataFrame (DatetimeIndex, includes 'close')
    반환: [(tkr, s, e, raw_score, {'corr':..,'lag':..,'hybrid':..}), ...] 하이브리드 점수 내림차순
    """
    from .scoring import best_lag_corr, to_similarity, hybrid_score as _hyb  # self import safe

    enriched = []
    for (tkr, s, e, sc) in matches:
        dfm = loader(tkr)
        seg = _slice_window_exact(dfm, s, W)
        if seg.empty or len(seg) < W:
            continue
        m_close = seg["close"].to_numpy(float)
        # 피벗정규화 과거 %시계열
        y_cmp = _pivot_past_only(m_close)
        corr, lag = best_lag_corr(y_ref, y_cmp, lag_max=lag_max)
        sim01 = to_similarity(sc, is_cosine)
        hyb = _hyb(sim01, corr, lag, lag_max=lag_max, w_sim=w_sim, w_corr=w_corr, lag_pen=lag_pen)
        enriched.append((tkr, s, e, float(sc), {'corr': float(corr), 'lag': int(lag), 'hybrid': float(hyb)}))

    enriched.sort(key=lambda x: x[4]['hybrid'], reverse=True)
    return enriched[:topk] if topk else enriched

# --- 내부 헬퍼(로컬 복사): 외부 의존 없이 사용 가능하도록 최소화 ---
def _slice_window_exact(df, start, W: int):
    s = pd.to_datetime(start, errors="coerce")
    if s is None or pd.isna(s): return pd.DataFrame()
    s = s.normalize()
    left = df.index.searchsorted(s, side="left")
    left = max(0, min(left, len(df) - W))
    return df.iloc[left:left+W]

def _pivot_past_only(win_close: np.ndarray) -> np.ndarray:
    base = float(win_close[-1]) if np.isfinite(win_close[-1]) and win_close[-1] != 0 else 1e-6
    y = (np.asarray(win_close, float) / base - 1.0) * 100.0
    return y - y[-1]
