# pattern/api.py
from fastapi import FastAPI
from pydantic import BaseModel
import json
import numpy as np
from .indexer import INDEX_DIR
from .query import query
from .data_loader import load_parquet
from .features import add_features
from .windows import make_windows
from .embed_baseline import embed_window

app = FastAPI()

class ScanReq(BaseModel):
    target: str           # 티커
    window: int = 60
    topk: int = 3
    exclude_recent_days: int = 60
    rerank: bool = False
    fwd_days: int = 20

def get_close_series(ticker, s, e):
    df = load_parquet(ticker)
    df = df.loc[s:e]
    x = df["close"].values.astype(float)
    # 정규화(형태 비교 목적)
    x = (x - x.mean()) / (x.std() + 1e-8)
    return x.tolist()

@app.post("/scan")
def scan(req: ScanReq):
    # 타겟 최근 window 추출
    df = load_parquet(req.target)
    feat = add_features(df).tail(req.window + req.fwd_days + 1)
    X, y, meta = make_windows(feat, ["close","logv","ma5","ma20","ma60","ma120","rsi14","mfi14","stochK","atr14","px_over_ma20"],
                              W=req.window, step=1, fwd_days=req.fwd_days)
    if len(X) == 0: return {"ok":False, "error":"not enough data"}

    tgtW = X[-1]     # 가장 최근 윈도우
    tgt_meta = (str(meta[-1][0].date()), str(meta[-1][1].date()))

    # 메타 로드
    name = f"W{req.window}_retstats"
    raw_meta = json.loads((INDEX_DIR / f"{name}.meta.json").read_text(encoding="utf-8"))

    def provider(t,s,e): return get_close_series(t, s, e)

    out = query(tgtW, tgt_meta, name, raw_meta,
                topk=req.topk,
                exclude_recent_days=req.exclude_recent_days,
                rerank=req.rerank,
                series_provider=provider)

    return {
        "ok": True,
        "target": {"ticker": req.target, "start": tgt_meta[0], "end": tgt_meta[1]},
        "matches": out
    }
