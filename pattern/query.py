# pattern/query.py
import numpy as np, json, datetime as dt
from .embed_baseline import embed_window
from .indexer import search_faiss
from tslearn.metrics import dtw as dtw_distance
from fastdtw import fastdtw

def to_date(s): return dt.datetime.strptime(s, "%Y-%m-%d").date()

def rerank_dtw(target_close_seq, candidates_close_seq, topk=10, use_fast=True):
    # DTW로 정밀 재채점(선택)
    scores = []
    for i, seq in enumerate(candidates_close_seq):
        if use_fast:
            d,_ = fastdtw(target_close_seq, seq)
        else:
            d = dtw_distance(target_close_seq, seq)
        scores.append((i, d))
    scores.sort(key=lambda x: x[1])  # 거리 작은 순
    order = [i for i,_ in scores[:topk]]
    return order

def query(top_window_np, meta_top_window, index_name, raw_meta,
          topk=10, exclude_recent_days=60, rerank=False, series_provider=None):
    """
    top_window_np: (W,F) 타겟 윈도우
    meta_top_window: (start_date, end_date)
    raw_meta: list of (ticker, start, end) for index rows
    series_provider: callable(ticker, start, end) -> close array (재채점용)
    """
    q = embed_window(top_window_np)  # (D,)
    D,I = search_faiss(q[None,:], index_name, topk=topk*5)

    # 리크 방지 필터
    cutoff = dt.date.today() - dt.timedelta(days=exclude_recent_days)
    out = []
    for d, i in zip(D, I):
        t, s, e = raw_meta[int(i)]
        if to_date(e) >= cutoff:  # 최근 구간 제외
            continue
        out.append({"ticker":t, "start":s, "end":e, "dist":float(d), "idx":int(i)})
        if len(out) >= topk*2: break

    # 재채점(DTW) 옵션
    if rerank and series_provider is not None:
        tgt_close = series_provider(*meta_top_window)  # (W,)
        cseq = [series_provider(o["ticker"], o["start"], o["end"]) for o in out]
        order = rerank_dtw(tgt_close, cseq, topk=topk)
        out = [out[i] for i in order]
    else:
        out = out[:topk]

    return out
