# scripts/build_dashboard.py
# -*- coding: utf-8 -*-
"""
로컬 대시보드 HTML 생성 (인자 없이 실행)
- 이미지/CSV는 outputs/ 폴더의 산출물을 그대로 사용
- 진행바 표시/해제, 표-행 클릭 시 상세 비교차트 갱신
"""

from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

# ---------------- User Config ----------------
TARGET      = "005930"
WINDOW      = 60
TOPK        = 3
HORIZON     = 20

# 파일명 규칙(이전 스크립트들과 맞춤)
OVERLAY_MAIN      = f"overlay_local_{TARGET}_W{WINDOW}_K{TOPK}.png"
PRED_CURVE        = f"predict_curve_{TARGET}_W{WINDOW}_H{HORIZON}.png"
PRED_BAND         = f"predict_curve_band_{TARGET}_W{WINDOW}_H{HORIZON}.png"   # 있으면 추가로 노출
HISTO_PREFIX      = f"predict_hist_H{HORIZON}_{TARGET}_W{WINDOW}"              # 있으면 노출
MATCHES_CSV       = f"matches_{TARGET}_W{WINDOW}_K{TOPK}.csv"
MATCH_IMG_PATTERN = "overlay_local_{t}_match{rank}.png"  # rank=1..TOPK

# ---------------- Paths ----------------
ROOT     = Path(__file__).resolve().parents[1]
OUT_DIR  = ROOT / "outputs"
HTML_OUT = OUT_DIR / "dashboard.html"

# ---------------- Helpers ----------------
def _exists(p: Path) -> bool:
    return p.exists() and p.is_file()

def _img_tag(rel_path: str, alt: str) -> str:
    fp = OUT_DIR / rel_path
    if not _exists(fp):
        return f'<div class="img-missing">({alt} 이미지 없음)</div>'
    # cache-bust: ts 쿼리스트링
    ts = int(fp.stat().st_mtime)
    return f'<img id="{alt}" src="{rel_path}?v={ts}" alt="{alt}" loading="lazy"/>'

def _read_matches() -> list[dict]:
    csvp = OUT_DIR / MATCHES_CSV
    if not _exists(csvp):
        return []
    df = pd.read_csv(csvp)
    need = [c for c in ["rank","ticker","start","end","sim","dist"] if c in df.columns]
    df = df[need]
    rows = []
    for _, r in df.iterrows():
        rank = int(r["rank"])
        tkr  = str(r["ticker"])
        s    = str(r["start"])
        e    = str(r["end"])
        score = None
        if "sim" in df.columns and not pd.isna(r.get("sim")):
            score = f"sim={float(r['sim']):.6f}"
        elif "dist" in df.columns and not pd.isna(r.get("dist")):
            score = f"d={float(r['dist']):.4f}"
        rows.append({"rank":rank,"tkr":tkr,"s":s,"e":e,"score":score or ""})
    rows.sort(key=lambda x:x["rank"])
    return rows[:TOPK]

def main():
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    # 이미지 존재 여부 체크
    overlay_main_tag = _img_tag(OVERLAY_MAIN, "overlay_main")
    pred_curve_tag   = _img_tag(PRED_CURVE,   "forecast_curve")
    pred_band_tag    = _img_tag(PRED_BAND,    "forecast_band") if _exists(OUT_DIR / PRED_BAND) else ""

    # 히스토그램(옵션)
    hist_tags = []
    for suffix in ["", "_W60"]:  # 호환 여지(파일명 변형 대비)
        cand = OUT_DIR / f"{HISTO_PREFIX}{suffix}.png"
        if _exists(cand):
            ts = int(cand.stat().st_mtime)
            hist_tags.append(f'<img src="{cand.name}?v={ts}" alt="histogram" loading="lazy"/>')
    hist_html = "".join(hist_tags) if hist_tags else '<div class="img-missing">(히스토그램 없음)</div>'

    # 표 데이터(브라우저 파일 접근 제한 때문에 파이썬이 미리 굽는다)
    rows = _read_matches()
    rows_json = json.dumps(rows, ensure_ascii=False)

    # 상세 비교 기본 이미지(1위가 있으면 그걸로)
    if rows:
        first_match = MATCH_IMG_PATTERN.format(t=TARGET, rank=rows[0]["rank"])
        match_default_tag = _img_tag(first_match, "match_overlay")
    else:
        match_default_tag = '<div class="img-missing">match overlay</div>'

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>pattern-ai · 로컬 대시보드</title>
<style>
:root {{
  --bg:#0b0c10; --fg:#e9eef2; --muted:#9da7b3; --card:#101218; --border:#1a2030; --accent:#6ee7b7;
}}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Noto Sans KR',sans-serif; }}
.wrap {{ max-width:1280px; margin:0 auto; padding:22px 16px 44px; }}
h1 {{ margin:0 0 16px; font-size:24px; font-weight:800; }}
.badge {{ margin-left:10px; background:#1f2937; color:#cbd5e1; font-size:12px; padding:3px 8px; border-radius:999px; }}
.row {{ display:grid; gap:16px; }}
.row2 {{ grid-template-columns:1fr 1fr; }}
.row3 {{ grid-template-columns:1fr 1fr; }}
@media (min-width: 1100px) {{ .row3 {{ grid-template-columns:1.4fr 1fr; }} }}
.card {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:12px 12px; }}
.card h2 {{ font-size:16px; margin:0 0 10px; }}
.small {{ color:var(--muted); font-size:13px; }}
img {{ max-width:100%; height:auto; display:block; border-radius:10px; }}
.img-missing {{ color:var(--muted); font-size:13px; padding:18px 6px; }}
.topbar {{ display:flex; align-items:center; gap:10px; margin-bottom:10px; }}
input[type=text] {{ background:#0f1420; color:#e5edf6; border:1px solid #22304a; border-radius:10px; padding:10px 12px; outline:none; width:140px; }}
.btn {{ background:#1f2937; color:#e5edf6; border:1px solid #334155; padding:9px 12px; border-radius:10px; cursor:pointer; }}
.btn:hover {{ background:#263244; }}
.progress {{
  height:8px; background:#0e1624; border-radius:999px; overflow:hidden; position:relative; flex:1;
}}
.progress > .bar {{
  position:absolute; left:-40%; top:0; bottom:0; width:40%;
  background:linear-gradient(90deg,#34d399,#60a5fa);
  animation: slide 1.2s infinite ease-in-out;
}}
.progress.hidden {{ display:none; }}
@keyframes slide {{
  0% {{ left:-40%; }} 50% {{ left:60%; }} 100% {{ left:100%; }}
}}
table {{ width:100%; border-collapse: collapse; font-size:14px; }}
th, td {{ padding:10px 8px; border-bottom:1px solid #1d2536; text-align:left; }}
tbody tr {{}}
tbody tr:hover {{ background:#131a26; cursor:pointer; }}
.footer {{ margin-top:12px; color:var(--muted); font-size:12px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>pattern-ai · 로컬 대시보드 <span class="badge">W {WINDOW}</span></h1>

  <div class="topbar">
    <label class="small">티커</label>
    <input type="text" id="ticker" value="{TARGET}" />
    <button class="btn" id="btnRun">조회</button>
    <div class="progress hidden" id="prog"><div class="bar"></div></div>
  </div>

  <div class="row row2">
    <div class="card">
      <h2>유사 패턴 오버레이 (Top-{TOPK})</h2>
      {overlay_main_tag}
      <div class="small">타깃 과거: 실선 · 비교과거: 실선 · 비교미래: 점선 (t=0 연속)</div>
    </div>
    <div class="card">
      <h2>미래 변동 예측 곡선 (H={HORIZON})</h2>
      {pred_curve_tag}
      {pred_band_tag}
      <div class="small">예측선: 가중평균(소프트맥스 β), 밴드/히스토그램은 outputs 폴더에 있으면 자동 노출</div>
    </div>
  </div>

  <div class="row row3" style="margin-top:16px;">
    <div class="card">
      <h2>유사 패턴 Top-{TOPK}</h2>
      <table>
        <thead><tr><th>순위</th><th>티커</th><th>기간</th><th>유사도/거리</th></tr></thead>
        <tbody id="tblBody">
          <!-- 파이썬이 rows_json으로 채움 -->
        </tbody>
      </table>
      <div class="small">행 클릭 → 오른쪽 박스에 상세 비교차트 표시</div>
    </div>
    <div class="card">
      <h2>상세 비교차트</h2>
      <div id="detailBox">
        {match_default_tag}
      </div>
    </div>
  </div>

  <div class="card" style="margin-top:16px;">
    <h2>예측 변동 히스토그램</h2>
    {hist_html}
  </div>

  <div class="footer">생성시각 {now}  ·  경로 기준: <code>outputs/</code></div>
</div>

<script>
// 표 데이터 주입
const ROWS = {rows_json};

// 테이블 렌더
function renderTable() {{
  const tb = document.getElementById('tblBody');
  tb.innerHTML = '';
  if (!ROWS || ROWS.length === 0) {{
    const tr = document.createElement('tr');
    tr.innerHTML = '<td colspan="4" class="small">CSV 없음</td>';
    tb.appendChild(tr);
    return;
  }}
  for (const r of ROWS) {{
    const tr = document.createElement('tr');
    tr.dataset.rank = r.rank;
    tr.innerHTML = `<td>${{r.rank}}</td><td>${{r.tkr}}</td><td>${{r.s}} ~ ${{r.e}}</td><td>${{r.score}}</td>`;
    tr.addEventListener('click', () => showDetail(r.rank));
    tb.appendChild(tr);
  }}
}}
renderTable();

// 상세 비교 이미지 교체
function showDetail(rank) {{
  const box = document.getElementById('detailBox');
  const t = '{TARGET}';
  const name = `overlay_local_${{t}}_match${{rank}}.png`;
  const url = `./${{name}}?v=${{Date.now()}}`;
  const img = new Image();
  img.onload = () => {{
    box.innerHTML = '';
    img.style.maxWidth = '100%';
    img.style.borderRadius = '10px';
    box.appendChild(img);
  }};
  img.onerror = () => {{
    box.innerHTML = '<div class="img-missing">(상세 비교 이미지 없음)</div>';
  }};
  img.src = url;
}}

// 진행바 제어
const prog = document.getElementById('prog');
function startProgress() {{ prog.classList.remove('hidden'); }}
function stopProgress()  {{ prog.classList.add('hidden'); }}

// 조회 버튼: 새로 생성된 파일을 바로 보이도록 이미지 src에 캐시버스터만 갱신
document.getElementById('btnRun').addEventListener('click', () => {{
  startProgress();
  const imgs = document.querySelectorAll('img');
  let remain = imgs.length;
  if (remain === 0) {{ stopProgress(); return; }}
  imgs.forEach(img => {{
    const base = img.src.split('?')[0];
    img.onload = () => {{ if (--remain === 0) stopProgress(); }};
    img.onerror = () => {{ if (--remain === 0) stopProgress(); }};
    img.src = base + '?v=' + Date.now();
  }});
}});
</script>
</body>
</html>
"""
    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"[ok] dashboard → {HTML_OUT}")

if __name__ == "__main__":
    main()
