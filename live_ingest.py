import json
import numpy as np
import faiss
import hashlib
import threading
import os
from datetime import datetime, timedelta
from pathlib import Path
from sentence_transformers import SentenceTransformer
from fetchers import fetch_papers, fetch_arxiv_papers

BASE = Path(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = str(BASE / "faiss.index")
META_PATH = str(BASE / "metadata.json")
SEEN_PATH = str(BASE / "seen_hashes.json")
EMBEDDING_DIM = 384

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
_write_lock = threading.Lock()
HASH_TTL_DAYS = 90


def get_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def load_seen() -> dict:
    if Path(SEEN_PATH).exists():
        with open(SEEN_PATH) as f:
            data = json.load(f)
            if isinstance(data, list):
                return {h: datetime.now().isoformat() for h in data}
            return data
    return {}


def save_seen(seen: dict):
    cutoff = datetime.now() - timedelta(days=HASH_TTL_DAYS)
    pruned = {h: ts for h, ts in seen.items() if datetime.fromisoformat(ts) > cutoff}
    with open(SEEN_PATH, "w") as f:
        json.dump(pruned, f)


def build_fresh_index(docs: list):
    """Called on first run — builds index from scratch."""
    print(f"\nBuilding fresh index from {len(docs)} papers...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    embeddings = model.encode(
        [d["text"] for d in docs],
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=32
    )
    index.add(np.array(embeddings, dtype="float32"))
    with _write_lock:
        faiss.write_index(index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False)
    today = datetime.now().isoformat()
    save_seen({get_hash(d["text"]): today for d in docs})
    print(f"Done. {len(docs)} papers indexed.")
    print(f"Written to {INDEX_PATH}")


def update_index(new_docs: list):
    """Called for weekly incremental updates."""
    if not new_docs:
        print("No documents to add.")
        return

    seen = load_seen()
    fresh = [d for d in new_docs if get_hash(d["text"]) not in seen]

    if not fresh:
        print("All documents already indexed.")
        save_seen(seen)
        return

    print(f"\nAdding {len(fresh)} new papers...")
    embeddings = model.encode(
        [d["text"] for d in fresh],
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=32
    )

    with _write_lock:
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, encoding="utf-8") as f:
            metadata = json.load(f)
        index.add(np.array(embeddings, dtype="float32"))
        metadata.extend(fresh)
        faiss.write_index(index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)

    today = datetime.now().isoformat()
    seen.update({get_hash(d["text"]): today for d in fresh})
    save_seen(seen)
    print(f"Index updated. Total papers: {len(metadata)}")


def run_update():
    """Weekly update — fetches last 7 days from arXiv."""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Weekly arXiv update...")
    docs = fetch_papers()
    update_index(docs)


if __name__ == "__main__":
    raw_path = BASE / "arxiv_papers_raw.json"

    if raw_path.exists():
        # Build from pre-fetched file (run fetchers.py first)
        print(f"Loading from {raw_path}...")
        with open(raw_path, encoding="utf-8") as f:
            docs = json.load(f)
        print(f"Loaded {len(docs)} papers")
        build_fresh_index(docs)
    elif not Path(INDEX_PATH).exists():
        # No index and no raw file — fetch and build
        print("No index found. Fetching from arXiv (90 days)...")
        docs = fetch_arxiv_papers(days_back=90, max_per_category=500)
        build_fresh_index(docs)
    else:
        # Index exists — run weekly update
        print("Index exists. Running incremental update...")
        run_update()