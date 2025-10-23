# pattern/embed_cnn.py (요약)
import torch, torch.nn as nn, numpy as np

class ConvEncoder(nn.Module):
    def __init__(self, in_ch, hid=64, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hid, 5, padding=2), nn.ReLU(),
            nn.Conv1d(hid, hid, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # (B,hid,1)
        )
        self.fc = nn.Linear(hid, out_dim)
    def forward(self, x):  # x: (B, F, W)
        h = self.net(x).squeeze(-1)   # (B, hid)
        z = self.fc(h)                 # (B, out_dim)
        return nn.functional.normalize(z, dim=1)

def embed_windows_cnn(model, X):  # X: (N,W,F)
    with torch.no_grad():
        t = torch.tensor(X).permute(0,2,1).float()  # (N,F,W)
        z = model(t).cpu().numpy().astype("float32")
    return z
