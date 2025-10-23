# pattern/features.py

# -*- coding: utf-8 -*-
"""
==============================================================================
Module: features.py
Author: MoneyRecipe (pattern-ai)
Purpose:
    전처리 단계에서 주가 데이터(OHLCV)에 각종 보조지표(technical indicators)를
    자동 계산하여, 패턴 임베딩 및 유사 차트 탐색 모델의 입력으로 사용할 수 있는
    확장된 피처셋(Feature Set)을 생성한다.

Core Function:
    add_features(df: pd.DataFrame) -> pd.DataFrame

Inputs:
    df : 일별 주가 데이터 (DataFrame)
        반드시 다음 컬럼을 포함해야 함:
        ['open', 'high', 'low', 'close', 'volume']

Outputs:
    DataFrame
        원본 OHLCV + 아래 기술적 지표 및 변환 컬럼을 포함
        ├─ ma5, ma20, ma60, ma120        : 단기~장기 이동평균선
        ├─ rsi14                         : RSI(14) — 상대강도지수
        ├─ mfi14                         : MFI(14) — 거래량 가중 RSI
        ├─ stochK                        : Stochastic %K (14,3)
        ├─ atr14                         : ATR(14) — 평균진폭(변동성)
        ├─ px_over_ma20                  : 20일선 대비 상대가격(%) = close/ma20 - 1
        └─ logv                          : log(1 + volume) — 거래량 로그스케일

Notes:
    - 내부적으로 ta 라이브러리(technical analysis) 사용
    - 결측치(NaN)는 dropna() 처리 후 반환
    - 표준화(z-score)는 전체 구간 기준이 아닌, 윈도우 내 정규화 단계에서 수행됨
    - 본 모듈은 build_index.py 에서 add_features()로 호출됨

Example:
    >>> from pattern.features import add_features
    >>> import pandas as pd
    >>> df = pd.read_csv("005930.csv")        # 삼성전자 일봉
    >>> df_feat = add_features(df)
    >>> print(df_feat.tail())

Version:
    v1.2 — 2025-10-11 (includes ta-based RSI/MFI/Stoch/ATR/log-volume)
==============================================================================
"""
"""

import pandas as pd
import numpy as np
import ta  # technical analysis

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # 이동평균
    for w in [5,20,60,120]:
        out[f"ma{w}"] = out["close"].rolling(w).mean()
    # RSI/MFI/Stoch
    out["rsi14"] = ta.momentum.RSIIndicator(out["close"], window=14).rsi()
    out["mfi14"] = ta.volume.MFIIndicator(out["high"], out["low"], out["close"], out["volume"], window=14).money_flow_index()
    stoch = ta.momentum.StochasticOscillator(out["high"], out["low"], out["close"], window=14, smooth_window=3)
    out["stochK"] = stoch.stoch()
    # ATR(변동성)
    out["atr14"] = ta.volatility.AverageTrueRange(out["high"], out["low"], out["close"], window=14).average_true_range()
    # 상대화 특징(가격 대비)
    out["px_over_ma20"] = out["close"] / out["ma20"] - 1
    # 거래량 로그스케일
    out["logv"] = np.log1p(out["volume"])
    # 정규화(피처별 z-score). 시계열 단위로 학습할 때는 윈도우 내부 z가 더 적합하므로, 여기서는 원본 유지
    return out.dropna()
"""

# pattern/features.py

import numpy as np
import pandas as pd
# 기존에 import ta 하셨다면 유지해도 되지만 ATR은 더이상 ta에 의존하지 않습니다.

def _safe_atr(series_high: pd.Series,
              series_low: pd.Series,
              series_close: pd.Series,
              window: int = 14) -> pd.Series:
    """윈도우 미충족 시 NaN을 유지하는 안전 ATR(단순 이동평균 버전)."""
    h = pd.to_numeric(series_high, errors="coerce")
    l = pd.to_numeric(series_low,  errors="coerce")
    c = pd.to_numeric(series_close, errors="coerce")

    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)

    # Wilder EMA 대신 단순 이동평균(SMA). 필요시 .ewm(alpha=1/window, adjust=False).mean()로 교체 가능.
    atr = tr.rolling(window=window, min_periods=window).mean()
    return atr


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: DatetimeIndex, columns include: open, high, low, close, volume
    반환: 원본 컬럼 + 아래 파생
      - logv, ma5, ma20, ma60, ma120, rsi14, mfi14, stochK, atr14, px_over_ma20
    """
    out = df.copy()

    # --------- 기본 유효성 & 정리 ---------
    need = ["open","high","low","close","volume"]
    for c in need:
        if c not in out.columns:
            raise ValueError(f"[features] missing column: {c}")

    # 무한/비정상값 정리
    out[need] = out[need].replace([np.inf, -np.inf], np.nan)

    # --------- 파생 계산 ---------
    # 거래량 로그
    out["logv"] = np.log1p(out["volume"].clip(lower=0))

    # 이동평균
    out["ma5"]   = out["close"].rolling(5,  min_periods=5).mean()
    out["ma20"]  = out["close"].rolling(20, min_periods=20).mean()
    out["ma60"]  = out["close"].rolling(60, min_periods=60).mean()
    out["ma120"] = out["close"].rolling(120, min_periods=120).mean()

    # RSI(14)
    delta = out["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    roll_up   = gain.rolling(14, min_periods=14).mean()
    roll_down = loss.rolling(14, min_periods=14).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    out["rsi14"] = 100 - (100 / (1 + rs))

    # MFI(14)
    tp = (out["high"] + out["low"] + out["close"]) / 3.0
    raw_mf = tp * out["volume"]
    pmf = np.where(tp > tp.shift(1), raw_mf, 0.0)
    nmf = np.where(tp < tp.shift(1), raw_mf, 0.0)
    pmf = pd.Series(pmf, index=out.index)
    nmf = pd.Series(nmf, index=out.index)
    pmf_sum = pmf.rolling(14, min_periods=14).sum()
    nmf_sum = nmf.rolling(14, min_periods=14).sum()
    mfr = pmf_sum / nmf_sum.replace(0, np.nan)
    out["mfi14"] = 100 - (100 / (1 + mfr))

    # Stochastic %K (14)
    ll = out["low"].rolling(14, min_periods=14).min()
    hh = out["high"].rolling(14, min_periods=14).max()
    out["stochK"] = (out["close"] - ll) / (hh - ll) * 100.0

    # ✅ 안전 ATR(14) — ta.AverageTrueRange 대신 사용
    out["atr14"] = _safe_atr(out["high"], out["low"], out["close"], window=14)

    # 가격/MA20 비율
    out["px_over_ma20"] = out["close"] / out["ma20"]

    # 후처리: 앞단 결측은 그대로 두고, 뒤는 ffill/bfill로 “잘못된 채우기”를 하지 않음.
    return out
