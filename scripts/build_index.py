# scripts/build_index.py
# -*- coding: utf-8 -*-
"""
Run 버튼만 누르면 인덱스 빌드.
- 유니버스 선택은 파일 상단 토글로 변경
- 로컬 ohlcv 없으면 (옵션) 자동 fetch
- (옵션) features 캐시 저장
- 84D 임베딩 + FAISS 인덱스 (신형 meta 포맷)

결과:
  index/{INDEX_NAME}.idx
  index/{INDEX_NAME}.meta.json   # {"normalized": true/false, "metric": "...", "items": [...]}
"""
from __future__ import annotations

from pathlib import Path
import sys
import traceback
import numpy as np
import pandas as pd

# ==========================
# 🔧 USER TOGGLES (여기만 바꿔서 사용)
# ==========================
UNIVERSE_MODE = "KOSPI"   # "LIST" | "KOSPI" | "KOSDAQ" | "ALL"
LIST_TICKERS  = [         # UNIVERSE_MODE="LIST"일 때만 사용
    "005930","000660","006400","035420","035720","051910","068270","005380","000270","012330",
    "015760","096770","066570","028260","005490","086790","010950","034220","105560","000810",
    "055550","032830","017670","003550","018260","090430","011170","097950","034730","011780",
    "000120","003670","036570","251270","011200","010130","004020","006800","009540","042660",
    "267250","028050","241560","047810","271560","316140","005830","086520","247540","091990",
]

AUTO_FETCH_IF_MISSING = True     # 로컬에 ohlcv 없으면 pykrx로 가져와 저장
WRITE_FEATURES_CACHE = True      # data/features/{ticker}.parquet 같이 저장

WINDOW    = 60
STEP      = 1
FWD_DAYS  = 20
VOL_MODE  = "zlog"               # "zlog" | "turnover_zlog" | "ratio"
INDEX_NAME = "W60_retstats_candle84"

# ✅ 지표 계산/윈도우 생성을 위한 최소 길이(이하 스킵)
#   - ma120, rsi14, atr14 등을 안정적으로 계산하려면 ~120+ 여유 필요
MIN_REQUIRED_ROWS = 130
# ==========================

# 프로젝트 루트 경로 보정
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATA_DIR    = ROOT_DIR / "data"
DATA_OHLCV  = DATA_DIR / "ohlcv"
DATA_FEAT   = DATA_DIR / "features"
INDEX_DIR   = ROOT_DIR / "index"

print("[cfg] ROOT_DIR  =", ROOT_DIR)
print("[cfg] INDEX_DIR =", INDEX_DIR)

# project modules
from pattern.data_loader import get_ticker_list, fetch_ohlcv
from pattern.features import add_features      # ← ATR14 안전 계산 버전 권장
from pattern.windows import make_windows
from pattern.embed_candle import embed_window_candle
from pattern.indexer import build_faiss

# 임베딩 입력 컬럼 (앞 5개 OHLCV 고정)
FEATURES = [
    "open","high","low","close","volume",
    "logv","ma5","ma20","ma60","ma120",
    "rsi14","mfi14","stochK","atr14","px_over_ma20",
]

# ---------- utils ----------
def _resolve_universe() -> list[str]:
    mode = UNIVERSE_MODE.upper().strip()
    if mode == "LIST":
        return [str(t).zfill(6) for t in LIST_TICKERS]
    if mode in ("KOSPI", "KOSDAQ", "ALL"):
        codes = get_ticker_list("ALL" if mode == "ALL" else mode)
        return [str(c).zfill(6) for c in codes]
    # fallback
    return [str(t).zfill(6) for t in LIST_TICKERS]

def _read_local_ohlcv(ticker: str) -> pd.DataFrame | None:
    """data/ohlcv/{ticker}.parquet 읽기 → DatetimeIndex 보장."""
    DATA_OHLCV.mkdir(parents=True, exist_ok=True)
    fp = DATA_OHLCV / f"{ticker}.parquet"
    if not fp.exists():
        return None

    df = pd.read_parquet(fp)
    # 중복 컬럼 제거
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # 'date' 승격 시도
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.set_index("date")
    else:
        # 호환: 혹시 다른 이름으로 저장된 경우
        for cand in ("Date", "datetime", "Datetime", "time", "Time"):
            if cand in df.columns:
                df[cand] = pd.to_datetime(df[cand], errors="coerce")
                df = df.set_index(cand)
                break

    if not isinstance(df.index, pd.DatetimeIndex):
        return None

    return df.sort_index()

def _save_local_ohlcv(ticker: str, df: pd.DataFrame):
    """data/ohlcv/{ticker}.parquet 저장 (항상 date 1개, index=False)."""
    DATA_OHLCV.mkdir(parents=True, exist_ok=True)
    out = df.copy()

    # 중복 컬럼 제거
    out = out.loc[:, ~out.columns.duplicated(keep="first")]

    # 인덱스가 무엇이든 'date' 컬럼을 1개 만들기(무조건 reset_index)
    out = out.reset_index()
    first_col = out.columns[0]
    if first_col != "date":
        out = out.rename(columns={first_col: "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # 저장
    (DATA_OHLCV / f"{ticker}.parquet").write_bytes(out.to_parquet(index=False))
    print(f"[ohlcv] saved {ticker}")

def _ensure_data(ticker: str) -> pd.DataFrame | None:
    df = _read_local_ohlcv(ticker)
    if df is not None and not df.empty:
        return df
    if not AUTO_FETCH_IF_MISSING:
        print(f"[skip] no local ohlcv: {ticker}")
        return None
    print(f"[fetch] {ticker} via KRX")
    df = fetch_ohlcv(ticker)
    if df is None or df.empty:
        print(f"[warn] empty: {ticker}")
        return None
    _save_local_ohlcv(ticker, df)
    return _read_local_ohlcv(ticker)

def _save_features_cache(ticker: str, feat: pd.DataFrame):
    """data/features/{ticker}.parquet 저장 (date 1개만, index=False, 견고화)."""
    if not WRITE_FEATURES_CACHE:
        return

    DATA_FEAT.mkdir(parents=True, exist_ok=True)
    df = feat.copy()

    # 0) 중복 컬럼 제거
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # 1) date 컬럼 정확히 1개만 만들기 (무조건 reset → 첫 컬럼명을 date로)
    df = df.reset_index()
    first_col = df.columns[0]
    if first_col != "date":
        df = df.rename(columns={first_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 2) date를 항상 첫 번째로
    cols = ["date"] + [c for c in df.columns if c != "date"]
    df = df[cols]

    # 3) 저장
    (DATA_FEAT / f"{ticker}.parquet").write_bytes(df.to_parquet(index=False))
    # print(f"[feat] saved {ticker}")

# ---------- main ----------
def main():
    tickers = _resolve_universe()
    print(f"[universe] {UNIVERSE_MODE}  count={len(tickers)}")

    all_emb, all_meta = [], []
    processed = skipped = 0

    for i, t in enumerate(tickers, start=1):
        try:
            df = _ensure_data(t)
            if df is None or df.empty:
                skipped += 1
                continue

            # ✅ 너무 짧은 시계열은 스킵 (지표 계산 안전 + 윈도우 확보)
            if len(df) < MIN_REQUIRED_ROWS:
                print(f"[skip] {t}: rows={len(df)} < MIN_REQUIRED_ROWS({MIN_REQUIRED_ROWS})")
                skipped += 1
                continue

            # 지표 생성 (features.add_features는 내부에서 NaN 처리/안전계산)
            feat = add_features(df)

            # 필수 컬럼 존재 확인 (없으면 스킵)
            missing = [c for c in FEATURES if c not in feat.columns]
            if missing:
                print(f"[skip] {t}: missing features={missing}")
                skipped += 1
                continue

            # NaN 구간 제거 후 캐시 저장
            feat = feat.dropna(subset=FEATURES)
            if feat.empty:
                print(f"[skip] {t}: features all-NaN after drop")
                skipped += 1
                continue

            _save_features_cache(t, feat)

            # 윈도우 확보(과거 W + 미래 FWD_DAYS)
            if len(feat) < WINDOW + FWD_DAYS:
                print(f"[skip] {t}: rows={len(feat)} < {WINDOW+FWD_DAYS}")
                skipped += 1
                continue

            X, y, meta = make_windows(
                feat, FEATURES, W=WINDOW, step=STEP, fwd_days=FWD_DAYS
            )
            if X is None or len(X) == 0:
                print(f"[skip] {t}: make_windows=0")
                skipped += 1
                continue

            # 임베딩 생성
            for w, (s, e) in zip(X, meta):
                z = embed_window_candle(w, vol_mode=VOL_MODE).astype("float32")
                all_emb.append(z)
                all_meta.append((t, str(s.date()), str(e.date())))

            processed += 1
            if i % 25 == 0:
                print(f"  ...{i}/{len(tickers)} processed={processed} emb={len(all_emb)}")

        except Exception as ex:
            skipped += 1
            print(f"[err ] {t}: {ex.__class__.__name__}: {ex}")
            # 디버깅 필요 시 주석 해제
            # traceback.print_exc()

    if not all_emb:
        raise RuntimeError("No embeddings created. Check data or FEATURES/min rows/calc errors.")

    emb = np.vstack(all_emb).astype("float32")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    build_faiss(emb, all_meta, INDEX_NAME)

    print(f"[built] {INDEX_NAME} shape={emb.shape}")
    print(f"[done ] tickers processed={processed}  skipped={skipped}")
    print(f"[files] {INDEX_DIR / (INDEX_NAME + '.idx')} / {INDEX_DIR / (INDEX_NAME + '.meta.json')}")

if __name__ == "__main__":
    main()
