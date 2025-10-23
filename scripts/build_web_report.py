# scripts/build_web_report.py
# -*- coding: utf-8 -*-
"""
predict_forward_curve.py + local_scan_plot.py 산출물을 한 페이지 HTML로 묶기
- 구성은 run_report.py를 '참조'
- 그래프는 개선된 최신 산출물 사용
- Run만 누르면 실행되도록 상단 토글에서 설정
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import base64, re
import pandas as pd

# ============ USER TOGGLES ============
TARGET = "028260"   # <- 여기서만 바꿔 쓰세요
W      = 60
TOPK   = 3
H      = 20
# =====================================

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

def _b64img(p: Path) -> str | None:
    if not p.exists(): return None
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")

def _fmt_pct(x, digits=2):
    try: return f"{float(x):+.{digits}f}%"
    except: return "-"

def _find_match_cards(target: str, W: int) -> list[Path]:
    return sorted(OUT_DIR.glob(f"overlay_local_{target}_match*.png"))

def _extract_match_title(p: Path) -> str:
    m = re.search(r"match(\d+)", p.stem, re.I)
    idx = m.group(1) if m else "?"
    return f"대표 사례 {idx}"

def build_html(target: str, W: int, TOPK: int, H: int) -> str:
    overlay_main = OUT_DIR / f"overlay_local_{target}_W{W}_K{TOPK}.png"
    forecast_png = OUT_DIR / f"forecast_{target}_W{W}_H{H}.png"
    forecast_csv = OUT_DIR / f"forecast_{target}_W{W}_H{H}.csv"

    overlay_b64  = _b64img(overlay_main)
    forecast_b64 = _b64img(forecast_png)
    match_cards  = _find_match_cards(target, W)

    summary_rows = []
    if forecast_csv.exists():
        df = pd.read_csv(forecast_csv)
        for t in (5, 10, 20):
            row = df[df["t"] == t]
            if not row.empty:
                r = row.iloc[0]
                summary_rows.append({
                    "t": int(r["t"]),
                    "mean_weighted": _fmt_pct(r["mean_weighted"]),
                    "median": _fmt_pct(r["median"]),
                    "p25": _fmt_pct(r["p25"]),
                    "p75": _fmt_pct(r["p75"]),
                    "p10": _fmt_pct(r["p10"]),
                    "p90": _fmt_pct(r["p90"]),
                })

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    cards_html = ""
    if overlay_b64:
        cards_html += f"""
        <div class="card">
          <div class="ctitle">유사 패턴 오버레이 (W={W}, K={TOPK})</div>
          <img src="{overlay_b64}" loading="lazy" alt="overlay"/>
          <div class="cdesc">타깃 과거: 실선 · 비교대상 과거: 실선 · 비교대상 미래: 점선 (t=0 연속)</div>
        </div>
        """
    if match_cards:
        for p in match_cards:
            b64 = _b64img(p)
            if not b64: continue
            title = _extract_match_title(p)
            cards_html += f"""
            <div class="card">
              <div class="ctitle">{title}</div>
              <img src="{b64}" loading="lazy" alt="{title}"/>
            </div>
            """

    forecast_section = ""
    if forecast_b64 or summary_rows:
        table_html = ""
        if summary_rows:
            rows = "\n".join(
                f"<tr><td>+{r['t']}일</td><td>{r['mean_weighted']}</td>"
                f"<td>{r['median']}</td><td>{r['p25']} ~ {r['p75']}</td>"
                f"<td>{r['p10']} ~ {r['p90']}</td></tr>"
                for r in summary_rows
            )
            table_html = f"""
            <div class="card">
              <div class="ctitle">예측 통계 요약 (가중 평균 / 중앙값 / 분위수)</div>
              <div class="table-wrap">
                <table>
                  <thead>
                    <tr><th>H</th><th>가중 평균</th><th>중앙값</th><th>P25–P75</th><th>P10–P90</th></tr>
                  </thead>
                  <tbody>
                    {rows}
                  </tbody>
                </table>
              </div>
              <div class="cdesc">가중 평균은 β-softmax(sim)에 기반한 이웃들의 기대값(%)</div>
            </div>
            """
        img_html = f"""
        <div class="card">
          <div class="ctitle">미래 예측 곡선 (H={H})</div>
          <img src="{forecast_b64}" loading="lazy" alt="forecast"/>
          <div class="cdesc">예측선: 점선(0에서 연속), 밴드: P10–P90 / P25–P75, 중앙값 라인 포함</div>
        </div>
        """ if forecast_b64 else ""
        forecast_section = f"""
        <h2>미래 예측</h2>
        <div class="row">
          {img_html}
          {table_html}
        </div>
        """

    html = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>pattern-ai Report · {target} · W{W}</title>
<style>
:root {{
  --bg:#0b0c10; --fg:#e9eef2; --muted:#98a2b3; --card:#111318; --border:#1d2230; --accent:#7cd4ff;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Noto Sans KR',sans-serif; }}
.wrap {{ max-width:1280px; margin:0 auto; padding:28px 20px 80px; }}
h1 {{ font-size:24px; margin:0 0 6px; }}
h2 {{ font-size:20px; margin:26px 0 12px; }}
.sub {{ color:var(--muted); font-size:13px; margin-bottom:18px; }}
.row {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.card {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:14px; }}
.ctitle {{ font-weight:700; margin-bottom:10px; }}
.cdesc {{ color:var(--muted); font-size:12px; margin-top:8px; }}
img {{ width:100%; height:auto; border-radius:10px; display:block; }}
.table-wrap {{ overflow:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th, td {{ border-bottom:1px solid var(--border); padding:8px 10px; text-align:center; white-space:nowrap; }}
th {{ color:#cdd6e3; }}
@media (max-width: 1024px) {{ .row {{ grid-template-columns:1fr; }} }}
.badge {{
  display:inline-block; font-size:12px; color:#0b1220; background:var(--accent);
  border-radius:999px; padding:3px 10px; margin-left:8px;
}}
</style></head>
<body><div class="wrap">
  <h1>pattern-ai 리포트 · {target} <span class="badge">W{W}</span></h1>
  <div class="sub">생성시각 {now} · 소스: local_scan_plot / predict_forward_curve · 개선된 시각화 룰 적용</div>

  <h2>유사 패턴 사례</h2>
  <div class="row">
    {cards_html if cards_html else '<div class="card">유사 패턴 이미지가 없습니다.</div>'}
  </div>

  {forecast_section if (forecast_b64 or summary_rows) else '<h2>미래 예측</h2><div class="card">예측 자료가 없습니다.</div>'}
</div></body></html>"""
    return html

def main():
    html = build_html(TARGET, W, TOPK, H)
    out_path = OUT_DIR / f"report_{TARGET}_W{W}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[ok] report → {out_path}")

if __name__ == "__main__":
    main()   # ← 인자 없이 실행
