# scripts/build_features_cache.py
# -*- coding: utf-8 -*-
"""
Build features cache parquet per ticker.
- Reads OHLCV parquet under data/ohlcv/{ticker}.parquet
- Computes indicators via features.add_features()
- Writes to data/features/{ticker}.parquet (ALWAYS with a 'date' column)
- Stores simple metadata (feature_version) for reproducibility.
"""

from pathlib import Path
import argparse, json
import pandas as pd
import numpy as np
from pattern.features import add_features

ROOT = Path(__file__).resolve().parents[1]
DATA_OHLCV = ROOT / "data" / "ohlcv"
DATA_FEAT  = ROOT / "data" / "features"
META_PATH  = DATA_FEAT / "_feature_meta.json"
FEATURE_VERSION = "v1.2-ta-14d"  # ← 지표셋/파라미터 바뀌면 이 값 변경

REQ_COLS = ["open", "high", "low", "close", "volume"]

def list_tickers_from_ohlcv():
    return [p.stem for p in DATA_OHLCV.glob("*.parquet")]

def _ensure_datetime_index(df: pd.DataFrame, src_path: Path) -> pd.DataFrame:
    """OHLCV를 DatetimeIndex로 정규화. 실패 시 명확히 에러."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()

    # 흔한 날짜 컬럼 우선
    for c in ["date", "Date", "datetime", "Datetime", "time", "Time"]:
        if c in df.columns:
            df = df.copy()
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df = df.set_index(c).sort_index()
            return df

    # datetime dtype 컬럼 자동 탐색
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            df = df.set_index(c).sort_index()
            return df

    raise ValueError(f"[bad ohlcv] {src_path} 에 DatetimeIndex/날짜 컬럼이 없습니다.")

def build_one(ticker: str):
    src = DATA_OHLCV / f"{ticker}.parquet"
    if not src.exists():
        print(f"[skip] no ohlcv: {ticker}")
        return False

    df = pd.read_parquet(src)
    df = _ensure_datetime_index(df, src)

    # 필수 컬럼 검증
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"[bad ohlcv] {src} 필수 컬럼 누락: {missing}")

    # NaN/inf 간단 정리
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    # OHLCV + 지표 계산 (features.py는 dropna()로 정리)
    out = add_features(df)

    # 저장: 'date' 컬럼을 반드시 포함하고 index=False
    out_to_save = out.copy()
    out_to_save.insert(0, "date", out_to_save.index.astype("datetime64[ns]"))
    out_to_save.reset_index(drop=True, inplace=True)

    DATA_FEAT.mkdir(parents=True, exist_ok=True)
    dst = DATA_FEAT / f"{ticker}.parquet"
    out_to_save.to_parquet(dst, index=False)
    print(f"[ok] {ticker} -> {dst.name} rows={len(out_to_save)}")
    return True

def main(tickers_csv: str | None, overwrite_meta: bool):
    if tickers_csv:
        tickers = [t.strip() for t in tickers_csv.split(",") if t.strip()]
    else:
        tickers = list_tickers_from_ohlcv()

    ok = 0
    for t in tickers:
        ok += build_one(t)

    if overwrite_meta or not META_PATH.exists():
        META_PATH.write_text(json.dumps({
            "feature_version": FEATURE_VERSION,
            "indicators": ["ma5","ma20","ma60","ma120","rsi14","mfi14","stochK","atr14","px_over_ma20","logv"],
            "notes": "features parquet always contains a 'date' column saved from DatetimeIndex"
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[meta] wrote {META_PATH}")

    print(f"[done] {ok} tickers processed")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=str, default=None, help="comma-separated tickers")
    ap.add_argument("--overwrite-meta", action="store_true")
    args = ap.parse_args()
    main(args.tickers, args.overwrite_meta)
