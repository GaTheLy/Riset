import arxiv
import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV"]

# arXiv batches submissions into daily announcements, which can delay a paper's
# appearance in search by up to 1 business day. We look back this many extra
# days on each run so we never miss papers at the boundary. ID dedup ensures
# we don't re-process anything caught by the overlap.
INDEXING_OVERLAP_DAYS = 1

STATE_FILE = Path(__file__).parent.parent / "data" / "arxiv_state.json"


@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    authors: list[str]
    published: datetime
    url: str
    categories: list[str]


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {"last_fetched_at": None, "seen_ids": []}
    return json.loads(STATE_FILE.read_text())


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def fetch_papers(
    query: str,
    categories: list[str] = None,
    max_results: int = 100,
    bootstrap_days: int = 7,
) -> list[Paper]:
    """
    Fetch papers from arXiv matching query within categories.

    On first run (no state file), fetches papers from the last `bootstrap_days`.
    On subsequent runs, fetches only papers newer than the last run, with a
    1-day lookback overlap to handle arXiv's indexing delay.

    Already-seen papers are skipped via ID deduplication. State is saved
    automatically after each run.
    """
    if categories is None:
        categories = CATEGORIES

    state = _load_state()
    seen_ids: set[str] = set(state["seen_ids"])
    now = datetime.now(timezone.utc)

    if state["last_fetched_at"] is None:
        from_date = now - timedelta(days=bootstrap_days)
    else:
        last_fetched = datetime.fromisoformat(state["last_fetched_at"])
        from_date = last_fetched - timedelta(days=INDEXING_OVERLAP_DAYS)

    # arXiv query date format: YYYYMMDDHHmmSS
    from_str = from_date.strftime("%Y%m%d%H%M%S")
    to_str = now.strftime("%Y%m%d%H%M%S")
    date_filter = f"submittedDate:[{from_str} TO {to_str}]"

    category_filter = " OR ".join(f"cat:{c}" for c in categories)
    full_query = f"({query}) AND ({category_filter}) AND ({date_filter})"

    client = arxiv.Client()
    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    new_papers: list[Paper] = []
    new_ids: list[str] = []
    skipped = 0

    for result in client.results(search):
        if result.entry_id in seen_ids:
            skipped += 1
            continue
        new_papers.append(Paper(
            id=result.entry_id,
            title=result.title,
            abstract=result.summary,
            authors=[a.name for a in result.authors],
            published=result.published,
            url=result.entry_id,
            categories=result.categories,
        ))
        new_ids.append(result.entry_id)

    state["last_fetched_at"] = now.isoformat()
    state["seen_ids"] = list(seen_ids | set(new_ids))
    _save_state(state)

    print(f"[arxiv] fetched {len(new_papers)} new papers, skipped {skipped} duplicates")
    return new_papers


if __name__ == "__main__":
    print("=== First run (bootstrapping last 7 days) ===")
    papers = fetch_papers("large language models", max_results=10, bootstrap_days=7)
    for p in papers:
        print(f"  • {p.published.strftime('%Y-%m-%d')} — {p.title[:70]}")

    print("\n=== Second run (simulating next week) ===")
    papers2 = fetch_papers("large language models", max_results=10, bootstrap_days=7)
    print(f"  Got {len(papers2)} new papers (all {10 - len(papers2)} already seen ones skipped)")
