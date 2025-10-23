# pattern/embed_baseline.py
import numpy as np

def zscore(a, axis=0, eps=1e-8):
    mu = a.mean(axis=axis, keepdims=True)
    sd = a.std(axis=axis, keepdims=True)
    return (a - mu) / (sd + eps)

def embed_window(window: np.ndarray, mode="returns+stats"):
    """
    window: (W, F)
    mode:
      - "returns": 종가 기준 로그수익률 z-score만 사용(형태)
      - "returns+stats": 수익률 + 요약통계특징
      - "multich": 모든 채널 z-score 후 평탄화
    """
    W, F = window.shape
    # 1) 첫 채널이 'close'라고 가정: 로그수익률
    close = window[:,0]
    r = np.diff(np.log(close + 1e-8))  # (W-1,)
    r = (r - r.mean()) / (r.std() + 1e-8)

    if mode == "returns":
        return r.astype("float32")

    if mode == "returns+stats":
        # 변동성, 기울기, 왜도/첨도 등 간단 통계
        vol = r.std()
        slope = np.polyfit(np.arange(len(r)), r, 1)[0]
        p25,p50,p75 = np.percentile(r,[25,50,75])
        feat = np.array([vol, slope, p25, p50, p75], dtype="float32")
        return np.concatenate([r.astype("float32"), feat])

    if mode == "multich":
        w = zscore(window, axis=0)      # (W,F)
        return w.reshape(-1).astype("float32")  # 평탄화

    raise ValueError("unknown mode")
