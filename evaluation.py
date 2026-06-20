"""
PaperPulse — Retrieval Evaluation Suite
========================================
Metrics computed:
  - Recall@K        (K = 1, 5, 10)
  - Precision@K     (K = 1, 5, 10)
  - MRR             (Mean Reciprocal Rank)
  - NDCG@K          (K = 5, 10)
  - Avg latency (ms)
  - Hybrid vs Semantic improvement

Ground truth:
  20 manually written queries, each with a list of expected
  keywords that MUST appear in the top-K results' title or abstract.
  This is "soft" ground truth — no need for exact paper IDs.

Run:
  python evaluation.py
  python evaluation.py --k 10 --save
"""

import json
import numpy as np
import faiss
import time
import argparse
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ── Ground truth benchmark ────────────────────────────────────
# Each entry: (query, [keywords that must appear in a relevant result])
# A retrieved result is "relevant" if ANY of its keywords appear
# in the result's title+abstract (case-insensitive).
# Keywords chosen to be specific enough to avoid false positives.
BENCHMARK = [
    (
        "How can language models use external knowledge sources?",
        ["retrieval-augmented", "retrieval augmented", "rag", "knowledge retrieval"]
    ),
    (
        "How can I fine tune large language models with limited GPU memory?",
        ["lora", "low-rank", "low rank adaptation", "parameter-efficient"]
    ),
    (
        "How do diffusion models generate images from noise?",
        ["diffusion model", "denoising", "score matching", "ddpm", "ddim"]
    ),
    (
        "How do models understand both images and text?",
        ["vision-language", "vision language", "multimodal", "vlm", "visual instruction"]
    ),
    (
        "How do language models reason step by step?",
        ["chain-of-thought", "chain of thought", "cot", "step-by-step reasoning"]
    ),
    (
        "How are chatbots aligned with human preferences?",
        ["rlhf", "reinforcement learning from human feedback", "reward model", "human preference"]
    ),
    (
        "How does self attention work in transformers?",
        ["self-attention", "transformer", "attention mechanism", "multi-head attention"]
    ),
    (
        "How are entities and relationships represented in knowledge graphs?",
        ["knowledge graph", "knowledge base", "graph embedding", "entity relation"]
    ),
    (
        "How can AI models be made smaller and faster for deployment?",
        ["quantization", "quantized", "int8", "int4", "model compression", "efficient inference"]
    ),
    (
        "How do AI agents use tools to complete tasks autonomously?",
        ["llm agent", "tool use", "tool-use", "autonomous agent", "agentic", "function calling"]
    ),
    (
        "How can machine learning be trained privately across multiple devices?",
        ["federated learning", "federated", "privacy-preserving", "distributed training"]
    ),
    (
        "How do graph neural networks classify nodes in a network?",
        ["graph neural", "gnn", "node classification", "graph convolutional"]
    ),
    (
        "Why do language models sometimes make up facts?",
        ["hallucination", "factuality", "factual", "faithfulness", "grounding"]
    ),
    (
        "How are objects detected and segmented in images?",
        ["object detection", "instance segmentation", "semantic segmentation", "yolo", "detr"]
    ),
    (
        "How can models learn new tasks without forgetting previous ones?",
        ["continual learning", "catastrophic forgetting", "lifelong learning", "incremental learning"]
    ),
    (
        "How can prompt engineering improve AI responses?",
        ["prompt engineering", "instruction tuning", "instruction following", "prompting"]
    ),
    (
        "How are realistic 3D scenes generated from images?",
        ["nerf", "neural radiance", "3d generation", "gaussian splatting", "3d reconstruction"]
    ),
    (
        "How do large language models generate computer code?",
        ["code generation", "code synthesis", "program synthesis", "copilot", "codex"]
    ),
    (
        "How are AI systems tested for safety and robustness?",
        ["safety", "alignment", "red teaming", "jailbreak", "adversarial", "harmful"]
    ),
    (
        "How do mixture of experts models scale efficiently?",
        ["mixture of experts", "moe", "sparse model", "expert routing"]
    ),
]


def is_relevant(result: dict, keywords: list) -> bool:
    """Soft relevance: keyword match in title or abstract."""
    text = (result.get("title","") + " " + result.get("text","")).lower()
    return any(kw.lower() in text for kw in keywords)


def recall_at_k(retrieved: list, keywords: list, k: int) -> float:
    """Fraction of top-k results that are relevant. Binary: 1 if any relevant, 0 otherwise."""
    top = retrieved[:k]
    return 1.0 if any(is_relevant(r, keywords) for r in top) else 0.0


def precision_at_k(retrieved: list, keywords: list, k: int) -> float:
    """Fraction of top-k results that are relevant."""
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if is_relevant(r, keywords)) / k


def reciprocal_rank(retrieved: list, keywords: list) -> float:
    """1/rank of first relevant result. 0 if none found."""
    for i, r in enumerate(retrieved, 1):
        if is_relevant(r, keywords):
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list, keywords: list, k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    top = retrieved[:k]
    dcg = sum(
        (1.0 if is_relevant(r, keywords) else 0.0) / np.log2(i + 2)
        for i, r in enumerate(top)
    )
    # Ideal DCG: all top results are relevant
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(top))))
    return dcg / idcg if idcg > 0 else 0.0


def run_semantic(model, index, metadata, query: str, k: int):
    """Pure semantic retrieval."""
    start = time.time()
    vec = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.array(vec, dtype="float32"), k)
    latency = (time.time() - start) * 1000
    results = []
    for s, i in zip(scores[0], indices[0]):
        if i != -1 and i < len(metadata):
            r = metadata[i].copy()
            r["score"] = float(s)
            r["sem_score"] = float(s)
            results.append(r)
    return results, latency


def run_hybrid(model, index, bm25, metadata, query: str, k: int):
    """Hybrid BM25 + semantic with RRF fusion."""
    fetch_k = k * 4
    start = time.time()

    vec = model.encode([query], normalize_embeddings=True)
    sem_scores, sem_indices = index.search(np.array(vec, dtype="float32"), fetch_k)
    sem_idx_list = [int(i) for i in sem_indices[0] if i != -1]
    sem_score_map = {int(i): float(s) for s, i in zip(sem_scores[0], sem_indices[0]) if i != -1}

    bm25_raw = bm25.get_scores(query.lower().split())
    bm25_top = np.argsort(bm25_raw)[::-1][:fetch_k].tolist()

    rrf = {}
    for rank, idx in enumerate(sem_idx_list):
        rrf[idx] = rrf.get(idx, 0) + 1 / (60 + rank + 1)
    for rank, idx in enumerate(bm25_top):
        rrf[idx] = rrf.get(idx, 0) + 1 / (60 + rank + 1)

    # Normalize RRF
    rrf_vals = list(rrf.values())
    lo, hi = min(rrf_vals), max(rrf_vals)
    norm = {idx: (s - lo) / (hi - lo) if hi > lo else 1.0 for idx, s in rrf.items()}

    sorted_rrf = sorted(norm.items(), key=lambda x: x[1], reverse=True)
    latency = (time.time() - start) * 1000

    results = []
    seen = set()
    for idx, rrf_score in sorted_rrf:
        if idx >= len(metadata):
            continue
        url = metadata[idx].get("url","")
        if url and url in seen:
            continue
        if url:
            seen.add(url)
        r = metadata[idx].copy()
        r["score"] = rrf_score
        r["sem_score"] = sem_score_map.get(idx, 0.0)
        results.append(r)
        if len(results) == k:
            break

    return results, latency


def evaluate(k_values=(1, 5, 10), save=True):
    BASE = Path(os.path.dirname(os.path.abspath(__file__)))
    index_path = BASE / "faiss.index"
    meta_path  = BASE / "metadata.json"

    if not index_path.exists():
        print("No faiss.index found. Run live_ingest.py first.")
        return

    print("Loading index and model...")
    model    = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    index    = faiss.read_index(str(index_path))
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)

    research = [m for m in metadata if m.get("type") == "research"]
    print(f"Index: {index.ntotal} total | {len(research)} arXiv papers")

    print("Building BM25...")
    bm25 = BM25Okapi([doc["text"].lower().split() for doc in metadata])

    max_k = max(k_values)

    # ── Per-query results ──────────────────────────────────────
    sem_metrics  = {k: {"recall":[], "precision":[], "ndcg":[]} for k in k_values}
    hyb_metrics  = {k: {"recall":[], "precision":[], "ndcg":[]} for k in k_values}
    sem_mrr, hyb_mrr = [], []
    sem_latencies, hyb_latencies = [], []

    print(f"\n{'Query':<45} {'SEM MRR':>8} {'HYB MRR':>8} {'Winner':>8}")
    print("─" * 75)

    for query, keywords in BENCHMARK:
        sem_results, sem_lat = run_semantic(model, index, metadata, query, max_k)
        hyb_results, hyb_lat = run_hybrid(model, index, bm25, metadata, query, max_k)

        sem_latencies.append(sem_lat)
        hyb_latencies.append(hyb_lat)

        s_mrr = reciprocal_rank(sem_results, keywords)
        h_mrr = reciprocal_rank(hyb_results, keywords)
        sem_mrr.append(s_mrr)
        hyb_mrr.append(h_mrr)

        for k in k_values:
            sem_metrics[k]["recall"].append(recall_at_k(sem_results, keywords, k))
            sem_metrics[k]["precision"].append(precision_at_k(sem_results, keywords, k))
            sem_metrics[k]["ndcg"].append(ndcg_at_k(sem_results, keywords, k))
            hyb_metrics[k]["recall"].append(recall_at_k(hyb_results, keywords, k))
            hyb_metrics[k]["precision"].append(precision_at_k(hyb_results, keywords, k))
            hyb_metrics[k]["ndcg"].append(ndcg_at_k(hyb_results, keywords, k))

        winner = "HYBRID ✓" if h_mrr > s_mrr else ("SEM ✓" if s_mrr > h_mrr else "TIE")
        q_short = query[:43] + ".." if len(query) > 43 else query
        print(f"{q_short:<45} {s_mrr:>8.3f} {h_mrr:>8.3f} {winner:>8}")

    print("─" * 75)

    # ── Summary table ──────────────────────────────────────────
    print(f"\n{'Metric':<30} {'Semantic':>12} {'Hybrid':>12} {'Δ':>8}")
    print("─" * 65)

    def row(label, s_val, h_val):
        delta = h_val - s_val
        sign = "+" if delta >= 0 else ""
        print(f"{label:<30} {s_val:>12.4f} {h_val:>12.4f} {sign}{delta:>7.4f}")

    row("MRR", np.mean(sem_mrr), np.mean(hyb_mrr))

    for k in k_values:
        row(f"Recall@{k}",
            np.mean(sem_metrics[k]["recall"]),
            np.mean(hyb_metrics[k]["recall"]))

    for k in k_values:
        row(f"Precision@{k}",
            np.mean(sem_metrics[k]["precision"]),
            np.mean(hyb_metrics[k]["precision"]))

    for k in [v for v in k_values if v >= 5]:
        row(f"NDCG@{k}",
            np.mean(sem_metrics[k]["ndcg"]),
            np.mean(hyb_metrics[k]["ndcg"]))

    print("─" * 65)
    row("Avg latency (ms)", np.mean(sem_latencies), np.mean(hyb_latencies))

    # ── Failure analysis ───────────────────────────────────────
    print(f"\n{'='*65}")
    print("FAILURE ANALYSIS — queries where both retrievers failed")
    print(f"{'='*65}")
    failures = []
    for i, (query, keywords) in enumerate(BENCHMARK):
        sem_results, _ = run_semantic(model, index, metadata, query, 5)
        hyb_results, _ = run_hybrid(model, index, bm25, metadata, query, 5)
        sem_hit = any(is_relevant(r, keywords) for r in sem_results[:5])
        hyb_hit = any(is_relevant(r, keywords) for r in hyb_results[:5])
        if not sem_hit and not hyb_hit:
            failures.append(query)
            print(f"  ✗ {query}")
            if hyb_results:
                print(f"    Best hybrid result: {hyb_results[0].get('title','')[:70]}")

    if not failures:
        print("  None — both retrievers found relevant results for all queries.")

    # ── Save results ───────────────────────────────────────────
    results_dict = {
        "index_stats": {
            "total_chunks": int(index.ntotal),
            "arxiv_papers": len(research),
            "embedding_dim": int(index.d),
        },
        "benchmark_size": len(BENCHMARK),
        "k_values": list(k_values),
        "semantic": {
            "mrr": round(float(np.mean(sem_mrr)), 4),
            **{f"recall_at_{k}": round(float(np.mean(sem_metrics[k]["recall"])), 4) for k in k_values},
            **{f"precision_at_{k}": round(float(np.mean(sem_metrics[k]["precision"])), 4) for k in k_values},
            **{f"ndcg_at_{k}": round(float(np.mean(sem_metrics[k]["ndcg"])), 4) for k in k_values if k >= 5},
            "avg_latency_ms": round(float(np.mean(sem_latencies)), 2),
        },
        "hybrid": {
            "mrr": round(float(np.mean(hyb_mrr)), 4),
            **{f"recall_at_{k}": round(float(np.mean(hyb_metrics[k]["recall"])), 4) for k in k_values},
            **{f"precision_at_{k}": round(float(np.mean(hyb_metrics[k]["precision"])), 4) for k in k_values},
            **{f"ndcg_at_{k}": round(float(np.mean(hyb_metrics[k]["ndcg"])), 4) for k in k_values if k >= 5},
            "avg_latency_ms": round(float(np.mean(hyb_latencies)), 2),
        },
        "hybrid_improvement": {
            "mrr_delta": round(float(np.mean(hyb_mrr) - np.mean(sem_mrr)), 4),
            "recall5_delta": round(float(np.mean(hyb_metrics[5]["recall"]) - np.mean(sem_metrics[5]["recall"])), 4) if 5 in k_values else None,
        },
        "failed_queries": failures,
    }

    if save:
        out = Path(os.path.dirname(os.path.abspath(__file__))) / "eval_results.json"
        with open(out, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nSaved to {out}")

    return results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaperPulse Retrieval Evaluation")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5, 10],
                        help="K values for Recall/Precision/NDCG (default: 1 5 10)")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save results to eval_results.json")
    args = parser.parse_args()
    evaluate(k_values=tuple(args.k), save=args.save)