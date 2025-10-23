# pattern/data_loader.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from pykrx import stock
import pandas as pd
from pathlib import Path
from datetime import date
import time

ROOT_DIR  = Path(__file__).resolve().parents[1]
DATA_DIR  = ROOT_DIR / "data"
DATA_OHLCV = DATA_DIR / "ohlcv"
DATA_FEAT  = DATA_DIR / "features"  # 참고: 피처 캐시는 여기 사용

def get_ticker_list(market: str = "ALL") -> list[str]:
    """KOSPI/KOSDAQ 전체 티커 목록."""
    if market == "ALL":
        return stock.get_market_ticker_list(market="KOSPI") + \
               stock.get_market_ticker_list(market="KOSDAQ")
    return stock.get_market_ticker_list(market=market)

def _yyyymmdd(s: str | None) -> str:
    if not s:
        return date.today().strftime("%Y%m%d")
    s = str(s).replace("-", "").replace("/", "")
    if len(s) == 8:
        return s
    # 가능한 케이스만 최소 보정
    # (예: '2015-01-01' → '20150101'는 위에서 이미 처리)
    return s

def fetch_ohlcv(ticker: str, start: str = "20150101", end: str | None = None,
                tries: int = 3, sleep_sec: float = 0.3) -> pd.DataFrame:
    """pykrx에서 OHLCV 수집 (KRX 기준). index=DatetimeIndex 보장."""
    start = _yyyymmdd(start)
    end   = _yyyymmdd(end)
    last_err = None
    for _ in range(max(1, tries)):
        try:
            df = stock.get_market_ohlcv_by_date(start, end, ticker)
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.rename(columns={
                "시가": "open", "고가": "high", "저가": "low",
                "종가": "close", "거래량": "volume"
            })
            df.index = pd.to_datetime(df.index)
            df = df[["open", "high", "low", "close", "volume"]].sort_index()
            return df
        except Exception as e:
            last_err = e
            time.sleep(sleep_sec)
    # 재시도 실패
    raise RuntimeError(f"[fetch_ohlcv] failed: {ticker} {start}~{end} :: {last_err}")

# ----------------------------
# 로컬 저장/로드 (OHLCV)
# ----------------------------
def save_parquet_ohlcv(ticker: str, df: pd.DataFrame):
    """data/ohlcv/{ticker}.parquet 로 저장 (항상 'date' 컬럼 포함, index=False)."""
    DATA_OHLCV.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out.insert(0, "date", out.index.astype("datetime64[ns]"))
        out = out.reset_index(drop=True)
    elif "date" not in out.columns:
        # index가 일반 인덱스면 date 넣어줌(가능하면 변환)
        out.insert(0, "date", pd.to_datetime(out.index, errors="coerce"))
        out = out.reset_index(drop=True)
    (DATA_OHLCV / f"{ticker}.parquet").write_bytes(out.to_parquet(index=False))

def load_parquet_ohlcv(ticker: str) -> pd.DataFrame:
    """data/ohlcv/{ticker}.parquet 읽기 → DatetimeIndex 보장."""
    fp = DATA_OHLCV / f"{ticker}.parquet"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_parquet(fp)
    # date 컬럼 승격 (규격화)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.set_index("date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        # 오래된 파일 호환
        cand = next((c for c in ["Date","datetime","Datetime","time","Time"] if c in df.columns), None)
        if cand:
            df[cand] = pd.to_datetime(df[cand], errors="coerce")
            df = df.set_index(cand)
    return df.sort_index()

def list_local_tickers() -> list[str]:
    """data/ohlcv 에 존재하는 로컬 티커 리스트."""
    DATA_OHLCV.mkdir(parents=True, exist_ok=True)
    return [p.stem for p in sorted(DATA_OHLCV.glob("*.parquet"))]

# ----------------------------
# 하위호환: 이전 API와 동일 시그니처
# (내부적으로 ohlcv 경로를 사용하게 리다이렉트)
# ----------------------------
def save_parquet(ticker: str, df: pd.DataFrame):
    """DEPRECATED: data/{ticker}.parquet → data/ohlcv/{ticker}.parquet 로 리다이렉트."""
    save_parquet_ohlcv(ticker, df)

def load_parquet(ticker: str) -> pd.DataFrame:
    """DEPRECATED: data/{ticker}.parquet → data/ohlcv/{ticker}.parquet에서 읽음."""
    return load_parquet_ohlcv(ticker)
