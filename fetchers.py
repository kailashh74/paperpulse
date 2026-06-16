import requests
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

ARXIV_CATEGORIES = [
    "cs.AI",
    "cs.CL",
    "cs.CV",
    "cs.LG",
    "cs.IR",
    "cs.RO",
    "stat.ML",
]

ARXIV_API = "http://export.arxiv.org/api/query"
NS = "{http://www.w3.org/2005/Atom}"


def _parse_arxiv_response(xml_text: str) -> list:
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall(f"{NS}entry"):
        try:
            arxiv_id_raw = entry.find(f"{NS}id").text.strip()
            arxiv_id = arxiv_id_raw.split("/abs/")[-1].split("v")[0].strip()
            title = entry.find(f"{NS}title").text.strip().replace("\n", " ")
            abstract = entry.find(f"{NS}summary").text.strip().replace("\n", " ")
            published_raw = entry.find(f"{NS}published").text.strip()
            pub_date = published_raw[:10]
            authors = [a.find(f"{NS}name").text.strip() for a in entry.findall(f"{NS}author")]
            authors_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
            all_cats = [c.get("term", "") for c in entry.findall(f"{NS}category")]
            if not abstract or len(abstract) < 50:
                continue
            papers.append({
                "text": f"{title}. {abstract}",
                "title": title,
                "authors": authors_str,
                "source": "arXiv",
                "page": "N/A",
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "arxiv_id": arxiv_id,
                "timestamp": pub_date,
                "type": "research",
                "citations": 0,
                "fields": all_cats,
            })
        except Exception as e:
            print(f"[PARSE ERROR] {e}")
    return papers


def fetch_arxiv_category(category: str, days_back: int = 7, max_results: int = 200) -> list:
    print(f"  [{category}] Fetching last {days_back} days...")
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
    today = datetime.now().strftime("%Y%m%d")
    all_papers = []
    start = 0
    batch_size = 100

    while len(all_papers) < max_results:
        params = {
            "search_query": f"cat:{category} AND submittedDate:[{cutoff}0000 TO {today}2359]",
            "start": start,
            "max_results": min(batch_size, max_results - len(all_papers)),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        try:
            response = requests.get(
                ARXIV_API, params=params, timeout=30,
                headers={"User-Agent": "PaperPulse/2.0"}
            )
            if response.status_code == 429:
                print(f"  [{category}] Rate limited, waiting 60s...")
                time.sleep(60)
                continue
            if response.status_code != 200:
                print(f"  [{category}] HTTP {response.status_code}")
                break
            batch = _parse_arxiv_response(response.text)
            if not batch:
                break
            all_papers.extend(batch)
            print(f"  [{category}] {len(all_papers)} papers so far...")
            start += len(batch)
            time.sleep(10)
        except Exception as e:
            print(f"  [{category}] Error: {e}")
            break

    print(f"  [{category}] Done: {len(all_papers)} papers")
    return all_papers


def fetch_arxiv_papers(days_back: int = 7, max_per_category: int = 200) -> list:
    print(f"\nFetching arXiv — last {days_back} days, up to {max_per_category} per category")
    all_papers = []
    seen_ids = set()
    for category in ARXIV_CATEGORIES:
        papers = fetch_arxiv_category(category, days_back=days_back, max_results=max_per_category)
        for p in papers:
            if p["arxiv_id"] not in seen_ids:
                seen_ids.add(p["arxiv_id"])
                all_papers.append(p)
        time.sleep(15)
    print(f"\nTotal unique papers: {len(all_papers)}")
    return all_papers


def fetch_semantic_scholar(query: str, max_results: int = 25) -> list:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "fields": "title,abstract,year,publicationDate,externalIds,url,citationCount",
        "limit": min(max_results, 25),
    }
    try:
        response = requests.get(url, params=params, timeout=20, headers={"User-Agent": "PaperPulse/2.0"})
        if response.status_code != 200:
            return []
        docs = []
        for paper in response.json().get("data", []):
            title = paper.get("title", "").strip()
            abstract = paper.get("abstract", "").strip()
            if not title or not abstract:
                continue
            arxiv_id = paper.get("externalIds", {}).get("ArXiv", "")
            docs.append({
                "text": f"{title}. {abstract}",
                "title": title,
                "authors": "",
                "source": "Semantic Scholar",
                "page": "N/A",
                "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else paper.get("url", ""),
                "arxiv_id": arxiv_id,
                "timestamp": paper.get("publicationDate", "")[:10],
                "type": "research",
                "citations": paper.get("citationCount", 0),
                "fields": []
            })
        return docs
    except Exception as e:
        print(f"Semantic Scholar Error: {e}")
        return []


def fetch_papers(query: str = "", max_results: int = 100):
    """Router — called by live_ingest and app."""
    if query:
        return fetch_semantic_scholar(query, max_results=min(max_results, 25))
    return fetch_arxiv_papers(days_back=7, max_per_category=200)


if __name__ == "__main__":
    # Bootstrap — run once to seed the index with 90 days of papers
    import json
    papers = fetch_arxiv_papers(days_back=90, max_per_category=500)
    with open("arxiv_papers_raw.json", "w") as f:
        json.dump(papers, f)
    print(f"Saved {len(papers)} papers to arxiv_papers_raw.json")