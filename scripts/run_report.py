# scripts/run_report.py
# -*- coding: utf-8 -*-
"""
End-to-end 리포트 생성 (Run만 누르면 실행)
- 이웃 검색 → 전망 밴드(5/10/20) → 히스토그램 → 대표 패널 Top-3 → HTML 한 장
- 실행 시작 시 산출물 정리(해당 티커/W 대상)
- 대표 패널 룰:
  · 타깃(원본): 과거만 실선
  · 비교대상: 과거=실선, 미래=점선 (t=0 연속)
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json, base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pattern.embed_candle import embed_window_candle
from pattern.indexer import search_faiss

# ============ USER TOGGLES ============
TARGET = "028260"      # <- 여기서만 바꿔 쓰세요
WINDOW = 60
HORIZONS = [5, 10, 20]
SIM_CUTOFF = 0.90
TOPN_SEARCH = 1500
MIN_SEP_DAYS = 8
K_REPR = 3

INDEX_NAME = "W60_retstats_candle84"
# =====================================

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "index"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

def _clean_old_outputs(target: str, W: int):
    for pat in [
        f"overlay_repr_{target}_W{W}_*.png",
        f"forecast_{target}_W{W}_*.png",
        f"hist_{target}_W{W}_*.png",
        f"report_{target}_W{W}*.html",
        f"matches_{target}_W{W}_*.csv",
    ]:
        for p in OUT_DIR.glob(pat):
            try: p.unlink()
            except: pass

def _read_feat(ticker: str) -> pd.DataFrame:
    fp = ROOT / "data" / "features" / f"{ticker}.parquet"
    df = pd.read_parquet(fp)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"[bad index] {fp}")
    df = df.sort_index().replace([np.inf,-np.inf], np.nan).ffill().bfill()
    return df

def _load_meta_doc(meta_path: Path):
    m = json.loads(meta_path.read_text(encoding="utf-8"))
    items = m["items"]
    metric = (m.get("metric") or "").lower()
    normalized = bool(m.get("normalized", metric == "cosine"))
    window = int(m.get("window", 0)) or None
    return {"items": items, "metric": metric, "normalized": normalized, "window": window}

def pivot_normalize(win_close: np.ndarray, fwd_close: np.ndarray | None = None):
    base = win_close[-1] if win_close[-1] != 0 else 1e-6
    y_p = (win_close / base - 1.0) * 100.0
    y_p -= y_p[-1]
    x_p = np.arange(-len(win_close)+1, 1, dtype=int)
    if fwd_close is None or len(fwd_close) == 0:
        return x_p, y_p, None, None
    y_f = (fwd_close / base - 1.0) * 100.0
    x_f = np.arange(1, len(fwd_close)+1, dtype=int)
    return x_p, y_p, x_f, y_f

def _b64img(path: Path) -> str | None:
    if not path.exists(): return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")

def run(target: str = TARGET, W: int = WINDOW):
    _clean_old_outputs(target, W)

    # 타깃 윈도우(과거 W) + 임베딩
    dft = _read_feat(target)
    seg_t = dft.iloc[-(W+HORIZONS[-1]) : -HORIZONS[-1]]
    ref_window = seg_t["close"].to_numpy(float)
    q_vec = embed_window_candle(
        dft[["open","high","low","close","volume"]].iloc[-(W+HORIZONS[-1]): -HORIZONS[-1]].to_numpy(np.float32),
        vol_mode="zlog"
    ).reshape(1,-1)
    ref_sdt, ref_edt = seg_t.index[0], seg_t.index[-1]

    # 이웃 검색
    scores, idxs = search_faiss(q_vec, INDEX_NAME, topk=max(TOPN_SEARCH, 100))
    scores = np.asarray(scores).reshape(-1)
    idxs = np.asarray(idxs).reshape(-1).astype(int)

    meta = _load_meta_doc(INDEX_DIR / f"{INDEX_NAME}.meta.json")
    items = meta["items"]
    metric = meta["metric"]
    is_cosine = (metric == "cosine") or meta["normalized"]

    # 후보 수집 + 자기자신/근접자기자신 제거
    cand = []
    for sc, idx in zip(scores, idxs):
        if is_cosine and sc < SIM_CUTOFF: break
        tkr, s, e = items[int(idx)] if isinstance(items[0], (list, tuple)) else (
            items[int(idx)]["ticker"], items[int(idx)]["start"], items[int(idx)]["end"])
        sdt, edt = pd.to_datetime(s), pd.to_datetime(e)
        if tkr == target:
            if (is_cosine and sc >= 0.999999) or abs((edt - ref_edt).days) < MIN_SEP_DAYS:
                continue
        cand.append((tkr, s, e, float(sc)))

    cand.sort(key=lambda x: x[3], reverse=True)
    reps = cand[:K_REPR]

    # 대표 패널 (룰: 타깃/비교 과거=실선, 비교 미래=점선 + t=0 연속)
    maxH = max(HORIZONS)
    for i, (tkr, s, e, sc) in enumerate(reps, start=1):
        dfm = _read_feat(tkr)
        sdt, edt = pd.to_datetime(s), pd.to_datetime(e)
        seg_c = dfm.loc[sdt:edt]
        close_c = seg_c["close"].to_numpy(float)

        pos_end = dfm.index.get_loc(edt)
        fwd_c = dfm["close"].to_numpy(float)[pos_end+1 : pos_end+1 + maxH]

        xt_p, yt_p, _, _   = pivot_normalize(ref_window, None)
        xc_p, yc_p, xf, yf = pivot_normalize(close_c, fwd_c)

        plt.figure(figsize=(10.8, 4.6)); ax = plt.gca(); ax.grid(alpha=0.3)
        ax.plot(xt_p, yt_p, lw=2.2, label=f"{target} (past)")
        line = ax.plot(xc_p, yc_p, "-", lw=2.0, label=f"{tkr} (past)")[0]
        if xf is not None and len(xf) > 0:
            xf = np.concatenate(([0], xf)); yf = np.concatenate(([0.0], yf))
            ax.plot(xf, yf, ":", lw=2.0, color=line.get_color(), label=f"{tkr} (fwd)")
        ax.axvline(0, color="k", ls=":", alpha=0.6)
        ax.set_xlabel("Days (pivot=0)"); ax.set_ylabel("Return from pivot (%)")
        ax.set_title(f"[{i}/{K_REPR}] {tkr} vs {target} | {s}~{e} | score={sc:.3f}")
        ax.legend()
        out = OUT_DIR / f"overlay_repr_{target}_W{W}_{i:02d}_{tkr}_{s}_{e}.png"
        plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()

    # 간단 HTML (대표 패널만)
    reps_imgs = sorted(OUT_DIR.glob(f"overlay_repr_{target}_W{W}_*.png"))
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    cards = []
    for i, p in enumerate(reps_imgs, start=1):
        b64 = _b64img(p)
        cards.append(f"""
        <div class="card">
          <div class="ctitle">대표 사례 {i}</div>
          <img src="{b64}" loading="lazy"/>
        </div>
        """)
    html = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>pattern-ai Report · {target} · W{W}</title>
<style>
:root {{ --bg:#0b0c10; --fg:#e9eef2; --muted:#98a2b3; --card:#111318; --border:#1d2230; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Noto Sans KR',sans-serif; }}
.wrap {{ max-width:1280px; margin:0 auto; padding:28px 20px 60px; }}
h1 {{ font-size:24px; margin:0 0 6px; }}
.sub {{ color:var(--muted); font-size:14px; margin-bottom:18px; }}
.row {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.card {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:14px; }}
.ctitle {{ font-weight:700; margin-bottom:8px; }}
img {{ width:100%; height:auto; border-radius:10px; display:block; }}
@media (max-width: 920px) {{ .row {{ grid-template-columns:1fr; }} }}
</style></head>
<body><div class="wrap">
  <h1>pattern-ai 리포트 · {target} <span style="font-size:12px;color:#98a2b3;">W{W}</span></h1>
  <div class="sub">생성시각 {now} · cutoff={SIM_CUTOFF} · N_topn={TOPN_SEARCH}</div>
  <div class="row">
    {''.join(cards) if cards else '<div class="card">대표 이미지가 없습니다.</div>'}
  </div>
</div></body></html>"""
    out_html = OUT_DIR / f"report_{target}_W{W}.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"[ok] report → {out_html}")

if __name__ == "__main__":
    run()   # ← 인자 없이 실행
