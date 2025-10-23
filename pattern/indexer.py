# pattern/indexer.py
import faiss, json, numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]     # pattern-ai/
INDEX_DIR = ROOT_DIR / "index"

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def _load_meta(name: str) -> dict:
    mpath = INDEX_DIR / f"{name}.meta.json"
    if not mpath.exists():
        # 구버전 호환: L2 거리, 비정규화로 간주
        return {"metric": "l2", "normalized": False}
    return json.loads(mpath.read_text(encoding="utf-8"))

def build_faiss(
    emb: np.ndarray,
    meta: list,
    name: str,
    *,
    metric: str = "cosine",          # "cosine" | "l2"
    hnsw_M: int = 32,
    efConstruction: int = 80
):
    """
    emb: (N, D) float array
    meta: 인덱스 항목 메타(list). 파일에 그대로 append하지 않고 통째로 기록.
    metric="cosine"이면 L2-normalize 후 Inner Product로 코사인 유사도.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    d = emb.shape[1]

    use_cosine = (metric.lower() == "cosine")
    metric_type = faiss.METRIC_INNER_PRODUCT if use_cosine else faiss.METRIC_L2

    xb = emb.astype("float32")
    if use_cosine:
        xb = _l2_normalize(xb)

    index = faiss.IndexHNSWFlat(d, hnsw_M, metric_type)
    index.hnsw.efConstruction = efConstruction
    index.add(xb)

    out_idx = str(INDEX_DIR / f"{name}.idx")
    faiss.write_index(index, out_idx)

    meta_doc = {
        "metric": "cosine" if use_cosine else "l2",
        "normalized": bool(use_cosine),
        "hnsw_M": hnsw_M,
        "efConstruction": efConstruction,
        "dim": int(d),
        "count": int(xb.shape[0]),
        "items": meta,                 # 사용자가 넘긴 항목 메타(리스트) 그대로 저장
    }
    (INDEX_DIR / f"{name}.meta.json").write_text(
        json.dumps(meta_doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def search_faiss(q: np.ndarray, name: str, topk: int = 50, efSearch: int = 64):
    """
    반환: (D_or_S, I)
      - metric=cosine: D_or_S는 'sim'(내적=코사인, 클수록 유사)
      - metric=l2    : D_or_S는 'dist'(작을수록 유사)
    """
    idx_path = str(INDEX_DIR / f"{name}.idx")
    meta_doc = _load_meta(name)
    index = faiss.read_index(idx_path)

    # HNSW 검색 정확도/속도 트레이드오프
    try:
        index.hnsw.efSearch = efSearch
    except Exception:
        pass

    xq = q.astype("float32")
    if meta_doc.get("normalized", False):
        xq = _l2_normalize(xq)

    D, I = index.search(xq, topk)
    return D[0], I[0]

def append_faiss(emb_new: np.ndarray, meta_new: list, name: str):
    """
    기존 인덱스에 새 벡터 추가.
    - meta.json을 읽어 metric/normalized 파라미터를 따라 정규화 여부를 일치시킴.
    """
    idx_path = str(INDEX_DIR / f"{name}.idx")
    meta_path = INDEX_DIR / f"{name}.meta.json"

    meta_doc = _load_meta(name)
    index = faiss.read_index(idx_path)

    xb = emb_new.astype("float32")
    if meta_doc.get("normalized", False):
        xb = _l2_normalize(xb)

    index.add(xb)
    faiss.write_index(index, idx_path)

    # 메타 업데이트
    old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if isinstance(old.get("items"), list):
        old["items"].extend(meta_new)
    else:
        old["items"] = meta_new
    old["count"] = int(old.get("count", 0) + xb.shape[0])

    meta_path.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
