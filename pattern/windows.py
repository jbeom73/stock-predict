# pattern/windows.py
import numpy as np
import pandas as pd

def make_windows(df_feat: pd.DataFrame, feature_cols, W=60, step=1, fwd_days=20):
    X, y, meta = [], [], []
    closes = df_feat["close"].values.astype(float)
    idx = df_feat.index

    arr = df_feat[feature_cols].values.astype(float)  # (N, F)
    N = len(df_feat)
    for s in range(0, N - W - fwd_days + 1, step):
        e = s + W
        window = arr[s:e]  # (W, F)
        # 라벨: fwd 수익률(종가 기준)
        future = closes[e + fwd_days - 1]
        now    = closes[e - 1]
        fwdret = (future / now) - 1.0
        X.append(window)
        y.append(fwdret)
        meta.append((idx[s], idx[e-1]))  # (시작일, 종료일)
    return np.array(X), np.array(y), meta
