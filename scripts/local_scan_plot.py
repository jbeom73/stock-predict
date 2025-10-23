# scripts/local_scan_plot.py
# -*- coding: utf-8 -*-

from pathlib import Path
import json, hashlib, platform, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pattern.embed_candle import embed_window_candle
from pattern.indexer import search_faiss

# ------------ Config ------------
INDEX_NAME = "W60_retstats_candle84"
TARGET = "005930"
WINDOW = 60
TOPK = 3
FWD_DAYS = 20
EXCLUDE_RECENT_DAYS = 0         # 가장 최근 W일 사용
MIN_SEP_DAYS = 10               # 동일 종목 창 최소 간격
SEARCH_BUFFER = 200             # 검색 버퍼

# 하이브리드 스코어 설정
USE_HYBRID_SCORE = True         # True면 하이브리드 정렬 적용
LAG_MAX = 5                     # ±LAG_MAX 일 내 래그 탐색
W_SIM = 0.4                     # 코사인/거리 유사도 가중
W_CORR = 0.6                    # 최대 상관 가중
LAG_PEN = 0.10                  # 래그 벌점 계수

ROOT = Path(__file__).resolve().parents[1]
DATA_FEAT = ROOT / "data" / "features"
INDEX_DIR = ROOT / "index"
OUT_DIR = ROOT / "outputs"; OUT_DIR.mkdir(exist_ok=True)

FEATURES_OHLCV = ["open","high","low","close","volume"]
SAVE_PER_MATCH = True
SAVE_MATCHES_CSV = True

# ------------ Utils ------------
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
    if not fp.exists(): raise FileNotFoundError(f"missing: {fp}")
    df = pd.read_parquet(fp)
    if not isinstance(df.index, pd.DatetimeIndex):
        col = next((c for c in ["date","Date","datetime","Datetime","time","Time"] if c in df.columns), None)
        if col is None:
            for c in df.columns:
                if np.issubdtype(df[c].dtype, np.datetime64): col=c; break
        if col is None: raise ValueError(f"{fp}: DatetimeIndex/날짜 컬럼 필요")
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
    return {"items": m["items"], "metric": metric, "normalized": bool(m.get("normalized", metric=="cosine"))}

def slice_window_exact(df: pd.DataFrame, start, W: int) -> pd.DataFrame:
    s = pd.to_datetime(start, errors="coerce")
    if pd.isna(s): return pd.DataFrame()
    left = df.index.searchsorted(s.normalize(), side="left")
    left = max(0, min(left, len(df)-W))
    return df.iloc[left:left+W]

def series_hash_close_norm(close: np.ndarray) -> str:
    base = close[0] if close.size and close[0]!=0 else 1e-6
    n = close / base
    return hashlib.md5(np.round(n.astype(np.float32), 4).tobytes()).hexdigest()

def dedup_matches_by_sep_and_shape(matches, loader, W):
    out, seen = [], set()
    for tkr, s, e, sc in matches:
        if any(tkr==t2 and abs((pd.to_datetime(s)-pd.to_datetime(s2)).days) < MIN_SEP_DAYS for t2,s2,*_ in out):
            continue
        dfm = loader(tkr); seg = slice_window_exact(dfm, s, W)
        if seg.empty or len(seg)<W: continue
        c = seg["close"].astype(float).to_numpy()
        h = series_hash_close_norm(c[:W])
        if h in seen: continue
        seen.add(h); out.append((tkr,s,e,sc))
    return out

def pivot_normalize(win_close: np.ndarray, fwd_close: np.ndarray | None = None):
    base = win_close[-1] if np.isfinite(win_close[-1]) and win_close[-1]!=0 else 1.0
    y_past = (np.asarray(win_close,float)/base - 1.0)*100.0
    y_past -= y_past[-1]
    W = len(win_close); x_past = np.arange(-W+1, 1, dtype=int)
    if fwd_close is None or len(fwd_close)==0: return x_past, y_past, None, None
    y_fwd = (np.asarray(fwd_close,float)/base - 1.0)*100.0
    x_fwd = np.arange(1, len(fwd_close)+1, dtype=int)
    return x_past, y_past, x_fwd, y_fwd

# ----- Hybrid scoring helpers -----
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

def _to_similarity(score: float, is_cosine: bool) -> float:
    if is_cosine: return float(score)
    d = max(0.0, float(score))
    return 1.0/(1.0+d)  # 0~1

def _hybrid(sim01: float, corr: float, lag: int, lag_max: int, w_sim: float, w_corr: float, lag_pen: float):
    return w_sim*sim01 + w_corr*((corr+1.0)/2.0) - (abs(lag)/max(1,lag_max))*lag_pen

# ------------ Main ------------
def main():
    _setup_korean_font()

    # 타깃 윈도우 & 임베딩
    dft = load_feat(TARGET)
    Wmat, _, t_edt = get_target_window_matrix(dft, WINDOW, EXCLUDE_RECENT_DAYS)
    q_vec = embed_window_candle(Wmat, vol_mode="zlog").reshape(1,-1)

    # 검색
    scores, idxs = search_faiss(q_vec, INDEX_NAME, topk=max(TOPK+SEARCH_BUFFER, 100))
    scores = np.asarray(scores).reshape(-1); idxs = np.asarray(idxs).astype(int)

    # 메타/유사도 타입
    meta = _load_meta_doc(INDEX_DIR / f"{INDEX_NAME}.meta.json")
    items = meta["items"]; is_cosine = (meta["metric"]=="cosine") or meta["normalized"]
    score_name = "sim" if is_cosine else "dist"

    # 후보 수집(자가/근접자기자신 제거)
    raw=[]
    for sc, ix in zip(scores, idxs):
        tkr, s, e = items[int(ix)] if isinstance(items[0], (list,tuple)) \
                    else (items[int(ix)]["ticker"], items[int(ix)]["start"], items[int(ix)]["end"])
        if tkr == TARGET:
            if (is_cosine and sc>=0.999999) or ((not is_cosine) and sc<=1e-9): continue
            if abs((pd.to_datetime(e)-t_edt).days) < MIN_SEP_DAYS: continue
        raw.append((tkr, s, e, float(sc)))

    # 중복 제거
    matches = dedup_matches_by_sep_and_shape(raw, load_feat, WINDOW)

    # ----- Hybrid 정렬 적용 -----
    if USE_HYBRID_SCORE and matches:
        # 타깃 과거 %시계열
        seg_t = dft.iloc[-WINDOW:] if EXCLUDE_RECENT_DAYS==0 else dft.iloc[-(WINDOW+EXCLUDE_RECENT_DAYS):-EXCLUDE_RECENT_DAYS]
        t_close = seg_t["close"].to_numpy(float)
        _, y_ref, _, _ = pivot_normalize(t_close, None)

        enriched=[]
        for (mt, ms, me, sc) in matches:
            dfm = load_feat(mt)
            seg = slice_window_exact(dfm, ms, WINDOW)
            if seg.empty or len(seg)<WINDOW: continue
            m_close = seg["close"].astype(float).to_numpy()
            _, y_cmp, _, _ = pivot_normalize(m_close[:WINDOW], None)

            corr, lag = _best_lag_corr(y_ref, y_cmp, LAG_MAX)
            sim01 = _to_similarity(sc, is_cosine)
            hyb = _hybrid(sim01, corr, lag, LAG_MAX, W_SIM, W_CORR, LAG_PEN)
            enriched.append((mt, ms, me, sc, {'corr':corr, 'lag':lag, 'hybrid':hyb}))

        enriched.sort(key=lambda x: x[4]['hybrid'], reverse=True)
        matches = enriched[:TOPK]
    else:
        matches.sort(key=lambda x: x[3], reverse=is_cosine)
        matches = matches[:TOPK]

    # 콘솔/CSV
    print("[matches]")
    for i,m in enumerate(matches,1):
        if len(m)>=5 and isinstance(m[4], dict):
            meta2=m[4]
            print(f"  {i}. {m[0]} {m[1]}~{m[2]}  {score_name}={m[3]:.6f}  corr={meta2['corr']:.3f}  lag={meta2['lag']:+d}  H={meta2['hybrid']:.3f}")
        else:
            tkr,s,e,sc = m
            print(f"  {i}. {tkr} {s}~{e}  {score_name}={sc:.6f}")

    if SAVE_MATCHES_CSV:
        csv_path = OUT_DIR / f"matches_{TARGET}_W{WINDOW}_K{TOPK}.csv"
        with open(csv_path,"w",newline="",encoding="utf-8") as f:
            w=csv.writer(f);
            hdr = ["rank","ticker","start","end",score_name,"corr","lag","hybrid"] if (matches and len(matches[0])>=5) else ["rank","ticker","start","end",score_name]
            w.writerow(hdr)
            for i,m in enumerate(matches,1):
                if len(m)>=5:
                    w.writerow([i, m[0], m[1], m[2], f"{m[3]:.6f}", f"{m[4]['corr']:.3f}", m[4]['lag'], f"{m[4]['hybrid']:.3f}"])
                else:
                    w.writerow([i, m[0], m[1], m[2], f"{m[3]:.6f}"])
        print(f"[ok] saved → {csv_path}")

    # 메인 오버레이(그림 로직 변경 없음)
    plt.figure(figsize=(11,6)); ax = plt.gca(); ax.margins(x=0.02,y=0.08); ax.grid(True, alpha=0.3)
    seg_t = dft.iloc[-WINDOW:] if EXCLUDE_RECENT_DAYS==0 else dft.iloc[-(WINDOW+EXCLUDE_RECENT_DAYS):-EXCLUDE_RECENT_DAYS]
    t_close = seg_t["close"].to_numpy(float)
    xt_p, yt_p, _, _ = pivot_normalize(t_close, None)
    ax.plot(xt_p, yt_p, lw=2.2, label=f"{TARGET} (past)")

    for m in matches:
        mt, ms, me, sc = m[:4]
        dfm = load_feat(mt); seg = slice_window_exact(dfm, ms, WINDOW)
        if seg.empty or len(seg)<WINDOW: continue
        m_close = seg["close"].astype(float).to_numpy()
        pos_end = dfm.index.get_loc(seg.index[-1])
        fwd = dfm["close"].astype(float).to_numpy()[pos_end+1: pos_end+1+FWD_DAYS]
        xp_p, yp_p, xp_f, yp_f = pivot_normalize(m_close[:WINDOW], fwd)

        lw = 1.6 + (max(0.0, (sc-0.85))*4.0 if is_cosine else 0.0)
        extra = ""
        if len(m)>=5 and isinstance(m[4], dict):
            extra = f" | corr={m[4]['corr']:.3f}, lag={m[4]['lag']:+d}, H={m[4]['hybrid']:.3f}"
        lbl = f"{mt} (past) {ms} ▶ {seg.index[0].date()}~{seg.index[-1].date()} ({'sim' if is_cosine else 'd'}={sc:.3f}){extra}"

        line, = ax.plot(xp_p, yp_p, lw=lw, label=lbl)
        if xp_f is not None and len(fwd)>0:
            xf = np.concatenate(([0], xp_f)); yf = np.concatenate(([0.0], yp_f))
            ax.plot(xf, yf, ":", lw=lw*0.9, color=line.get_color(), label=f"{mt} (future {len(fwd)}d)")

    ax.axvline(0, color="k", ls=":", alpha=0.6)
    ax.set_xlabel("Days (pivot=0)"); ax.set_ylabel("Return from pivot (%)")
    avg_disp = np.mean([m[3] for m in matches]) if matches else float('nan')
    ax.set_title(f"Overlay (W={WINDOW}, K={TOPK}) · {('Cosine' if is_cosine else 'L2')} · avg {score_name}={avg_disp:.3f}")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02,1)); plt.tight_layout(rect=[0,0,0.85,1])

    out = OUT_DIR / f"overlay_local_{TARGET}_W{WINDOW}_K{TOPK}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight"); print(f"[ok] saved → {out}")

    # 개별 비교 PNG(그림 로직 그대로)
    if SAVE_PER_MATCH and matches:
        for i,m in enumerate(matches,1):
            mt, ms, me, sc = m[:4]
            dfm = load_feat(mt); seg = slice_window_exact(dfm, ms, WINDOW)
            if seg.empty or len(seg)<WINDOW: continue
            m_close = seg["close"].astype(float).to_numpy()
            pos_end = dfm.index.get_loc(seg.index[-1])
            fwd = dfm["close"].astype(float).to_numpy()[pos_end+1: pos_end+1+FWD_DAYS]
            xp_p, yp_p, xp_f, yp_f = pivot_normalize(m_close[:WINDOW], fwd)

            plt.figure(figsize=(10,5)); ax2=plt.gca(); ax2.grid(True, alpha=0.3)
            ax2.plot(xt_p, yt_p, lw=2.2, label=f"{TARGET} (past)")
            line2, = ax2.plot(xp_p, yp_p, "-", lw=1.8, label=f"{mt} (past)")
            if xp_f is not None and len(fwd)>0:
                xf = np.concatenate(([0], xp_f)); yf = np.concatenate(([0.0], yp_f))
                ax2.plot(xf, yf, ":", lw=1.8, color=line2.get_color(), label=f"{mt} (future {len(fwd)}d)")
            ax2.axvline(0, color="k", ls=":", alpha=0.6)
            ax2.set_xlabel("Days (pivot=0)"); ax2.set_ylabel("Return from pivot (%)")
            ax2.set_title(f"{TARGET} vs {mt} | {ms} ▶ {seg.index[0].date()}~{seg.index[-1].date()} | {score_name}={sc:.3f}")
            ax2.legend()
            out_i = OUT_DIR / f"overlay_local_{TARGET}_match{i}.png"
            plt.tight_layout(); plt.savefig(out_i, dpi=160, bbox_inches="tight"); plt.close()
            print(f"[ok] saved → {out_i}")

if __name__ == "__main__":
    main()
