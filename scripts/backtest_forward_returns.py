# scripts/backtest_forward_returns.py
# -*- coding: utf-8 -*-
"""
Auto backtest of forward returns from the latest matches CSV.
- Just press Run: no CLI args needed.
- If no matches_*.csv exists, it tries to run local_scan_plot.main() to create one.
- Outputs:
    outputs/bt_<matches_stem>_detail.csv
    outputs/bt_<matches_stem>_summary.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
DATA_FEAT = ROOT / "data" / "features"
OUT_DIR   = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ---- Config ----
DEFAULT_HORIZONS = [5, 10, 20]     # R5/R10/R20
MATCHES_GLOB = "matches_*.csv"     # created by local_scan_plot.py

# ---------------- helpers ----------------
def _pick_latest_matches() -> Path | None:
    files = sorted(OUT_DIR.glob(MATCHES_GLOB), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def _ensure_matches_exists() -> Path:
    path = _pick_latest_matches()
    if path:
        print(f"[auto] using latest matches → {path.name}")
        return path

    # Try to generate via local_scan_plot.main()
    print("[auto] no matches CSV found; generating via local_scan_plot.main() ...")
    try:
        spec = importlib.util.spec_from_file_location("local_scan_plot", str(ROOT / "scripts" / "local_scan_plot.py"))
        mod = importlib.util.module_from_spec(spec)  # type: ignore
        assert spec and spec.loader
        spec.loader.exec_module(mod)                # type: ignore
        if hasattr(mod, "main"):
            mod.main()
        else:
            print("[warn] local_scan_plot.main() not found; please run local_scan_plot.py once.")
    except Exception as e:
        print(f"[warn] failed to run local_scan_plot: {type(e).__name__}: {e}")

    path = _pick_latest_matches()
    if not path:
        raise SystemExit("[error] No matches CSV found under outputs/. Run scripts/local_scan_plot.py first.")
    print(f"[auto] using latest matches → {path.name}")
    return path

def _try_load_feat(fp: Path) -> pd.DataFrame | None:
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    if "date" not in df.columns:
        raise RuntimeError(f"[bad features] {fp} must contain 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.set_index("date").sort_index()
    return df

def _load_feat(ticker: str) -> pd.DataFrame:
    """
    Load features parquet by ticker; robust to leading-zero loss.
    Tries {ticker}.parquet and {ticker.zfill(6)}.parquet.
    """
    t1 = str(ticker).strip()
    t2 = t1.zfill(6)
    for tk in (t1, t2):
        fp = DATA_FEAT / f"{tk}.parquet"
        df = _try_load_feat(fp)
        if df is not None:
            return df
    raise FileNotFoundError(f"[features missing] {DATA_FEAT / (t2 + '.parquet')}")

def _forward_returns(close: pd.Series, anchor_end, fwd=20):
    """
    anchor_end: window end date (from matches CSV). Use the next trading day as entry.
    Returns: (total_return, max_drawdown)
    """
    idx = close.index
    pos = idx.searchsorted(pd.to_datetime(anchor_end), side="right")
    if pos >= len(idx):
        return np.nan, np.nan
    end = min(pos + fwd, len(idx) - 1)
    path = close.iloc[pos:end+1].astype(float).to_numpy()
    if path.size < 2:
        return np.nan, np.nan
    ret = path[-1] / path[0] - 1.0
    peak = np.maximum.accumulate(path)
    mdd = (path / peak - 1.0).min()  # negative
    return float(ret), float(mdd)

# ---------------- main ----------------
def run(horizons=DEFAULT_HORIZONS):
    matches_path = _ensure_matches_exists()

    # Read matches with ticker as string (avoid losing leading zeros)
    mdf = pd.read_csv(matches_path, dtype={"ticker": str})

    # Normalize/guard columns
    score_col = "sim" if "sim" in mdf.columns else ("dist" if "dist" in mdf.columns else None)

    rows = []
    for _, r in mdf.iterrows():
        tkr = str(r["ticker"]).strip().zfill(6)   # ensure 6-digit ticker
        end = pd.to_datetime(r["end"])

        df = _load_feat(tkr)
        if "close" not in df.columns:
            continue
        close = df["close"]

        for H in horizons:
            ret, dd = _forward_returns(close, end, fwd=H)
            rows.append({
                "ticker": tkr,
                "end": end.date(),
                "horizon": H,
                "ret": ret,
                "mdd": dd,
                "score": r.get(score_col, np.nan) if score_col else np.nan
            })

    if not rows:
        raise SystemExit("[error] No forward-return rows computed. Check features and matches CSV.")

    out = pd.DataFrame(rows)
    stem = matches_path.stem.replace("matches_", "bt_")
    detail_path = OUT_DIR / f"{stem}_detail.csv"
    summary_path = OUT_DIR / f"{stem}_summary.csv"

    # Summary table
    summ = (out.groupby("horizon")
                .agg(n=("ret","count"),
                     win_rate=("ret", lambda x: float((x>0).mean())),
                     avg_ret=("ret","mean"),
                     med_ret=("ret","median"),
                     avg_mdd=("mdd","mean"),
                     p05_ret=("ret", lambda x: float(x.quantile(0.05))),
                     p95_ret=("ret", lambda x: float(x.quantile(0.95))))
                .reset_index())

    out.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summ.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"[ok] detail  → {detail_path}")
    print(f"[ok] summary → {summary_path}")
    print("\n=== SUMMARY ===")
    # Pretty print
    print(summ.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))

if __name__ == "__main__":
    # Just press Run
    run()
