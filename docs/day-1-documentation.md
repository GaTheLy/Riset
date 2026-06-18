# Day 1 Documentation — RAG Foundation

This is a living document. Each component gets a section as it's built. Use this to revisit decisions, understand the tradeoffs made, and prepare for interview questions about the architecture.

---

## Goal for Day 1

Build the retrieval backbone end-to-end:
- Ingest real papers and news articles
- Chunk, embed, and store them
- Retrieve relevant chunks via hybrid search (dense + sparse + RRF fusion)
- Answer a user query with cited sources

**Definition of done:** `python -m backend.api` runs, `POST /query` returns an answer with sources, and every step writes a JSONL log entry.

---

## Component 1 — arXiv Fetcher

**File:** `backend/ingestion/arxiv_fetcher.py`

### What it does

Periodically fetches recent AI/ML papers from the arXiv API and feeds them into the ingestion pipeline. It only fetches papers that haven't been seen before (incremental), and it writes a log entry after every run.

### Why it exists separately from the search tool

This was an important early design insight. There are two distinct use cases that look similar but are different:

| Concern | Where it lives | Has a query? | Purpose |
|---|---|---|---|
| **Ingestion fetcher** | `backend/ingestion/arxiv_fetcher.py` | No (optional) | Build the knowledge base — fetch *all* recent papers from target categories |
| **Search tool** | `backend/agent/` (Day 2) | Yes — required | Agent searches for papers on a *specific topic* at query time |

If you conflate these, your knowledge base only contains papers matching the query you happened to search for first. A RAG system's value comes from having a broad, pre-built corpus that the retrieval step can then search over.

---

### Design decisions

#### Sub-problem 1: How to avoid fetching old papers on every run

Three options were considered:

| Option | Approach | Pros | Cons |
|---|---|---|---|
| A | Always fetch the N most recent, sorted by date | Simple | Silently misses papers if N < papers published this period |
| **B (chosen)** | **Date-range query — store last fetch time, query only the new window** | Efficient, won't miss papers regardless of volume | Requires persisting state between runs |
| C | Fetch all from categories unconditionally | Guaranteed complete | Very inefficient as corpus grows; re-fetches everything every run |

**Choice: Option B** — date-range query with a persisted state file.

The arXiv API supports this via the `submittedDate` field in the query string:
```
submittedDate:[20260611000000 TO 20260618000000]
```

On each run, the fetcher reads `last_fetched_at` from a state file, calculates the new window, fetches, and writes the new timestamp back.

#### Sub-problem 1a: The arXiv indexing delay problem

arXiv doesn't make papers searchable instantly. It batches submissions and announces them the next business day (cut-off is 14:00 ET; announcement is ~20:00 ET). This means a paper submitted Monday might not be searchable until Tuesday evening.

**The risk:** if your last run was Monday 09:00 and you run again Tuesday 09:00, you'd query `[Mon 09:00 → Tue 09:00]`. Papers submitted Monday (announced Monday evening) *are* in the index by Tuesday morning — you'd catch those. But papers submitted Friday that get delayed by a weekend could slip through.

**Solution chosen: 1-day lookback overlap**

```
Normal window:    [last_fetched → now]
With overlap:     [last_fetched - 1 day → now]
```

The overlap is safe because ID deduplication (sub-problem 2) ensures any paper caught by the overlap that was already seen gets skipped. The overlap costs nothing in correctness — only a small extra API call over papers already in the state.

```
Timeline example (weekly run):

Week 1 run:  ├──────────────────────────────────┤
                                             ↑ saved as last_fetched_at

Week 2 run:                         ├──────────────────────────────────────┤
                                    ↑ last_fetched_at - 1 day (overlap zone)
                                                                        ↑ now

Papers in the overlap zone: already in seen_ids → skipped automatically
Papers in the new zone: new → processed and added to seen_ids
```

#### Sub-problem 2: How to avoid re-processing papers across runs

| Option | Approach | Pros | Cons |
|---|---|---|---|
| A | In-memory set within a single run | Simple | Lost on process exit — useless across runs |
| **B (chosen)** | **Persist seen IDs to a JSON state file** | Works across runs, O(1) lookup per paper | State file grows over time (~5MB after 2 years at 1K papers/week) |
| C | Derive seen IDs from the JSONL ingestion log | No extra file | Couples fetcher to log format; slow to rebuild as log grows |

**Choice: Option B** — a `seen_ids` set persisted in `backend/data/arxiv_state.json`.

Conceptually: on each run, we load the JSON array into a Python `set`. A `set` is backed by a hash table internally — checking `id in seen_ids` is O(1) regardless of how many IDs are stored. After fetching, the new IDs are unioned in and the set is saved back as a JSON array.

```
State file structure (backend/data/arxiv_state.json):
{
  "last_fetched_at": "2026-06-18T06:12:59+00:00",
  "seen_ids": [
    "http://arxiv.org/abs/2606.12345v1",
    "http://arxiv.org/abs/2606.23456v2",
    ...
  ]
}
```

> **Note:** `seen_ids` uses the full arXiv entry URL as the ID (e.g. `http://arxiv.org/abs/2606.19341v1`). The version suffix (`v1`, `v2`) means a revised paper gets a new ID and would be re-fetched. This is intentional — revisions can be significant enough to warrant re-ingestion.

---

### Full execution flow

```
fetch_papers(categories, query="", max_results=100, bootstrap_days=7)
        │
        ├─ 1. Load state.json
        │       ├─ last_fetched_at  (None if first run)
        │       └─ seen_ids         (loaded into a Python set)
        │
        ├─ 2. Calculate date window
        │       ├─ First run:  from = now - bootstrap_days
        │       └─ Later runs: from = last_fetched_at - 1 day (overlap)
        │
        ├─ 3. Build arXiv query string
        │       ├─ No query arg:  (cat:cs.AI OR cat:cs.LG OR ...) AND (submittedDate:[...])
        │       └─ With query:   (query) AND (cat:...) AND (submittedDate:[...])
        │
        ├─ 4. Call arXiv API → stream results
        │
        ├─ 5. For each result:
        │       ├─ id in seen_ids? → skip (duplicate)
        │       └─ else → append to new_papers, track new_id
        │
        ├─ 6. Save state.json
        │       ├─ last_fetched_at = now
        │       └─ seen_ids = old seen_ids ∪ new_ids
        │
        ├─ 7. Write JSONL log entry → backend/logs/ingestion_YYYYMMDD.jsonl
        │
        └─ 8. Return new_papers
```

---

### Data structures

**`Paper` dataclass** — the output type of this component:

| Field | Type | Source |
|---|---|---|
| `id` | `str` | `result.entry_id` — full URL, e.g. `http://arxiv.org/abs/2606.12345v1` |
| `title` | `str` | Paper title |
| `abstract` | `str` | Full abstract text |
| `authors` | `list[str]` | Author names |
| `published` | `datetime` | Submission date (timezone-aware) |
| `url` | `str` | Same as `id` — the abstract page URL |
| `categories` | `list[str]` | e.g. `["cs.CL", "cs.LG"]` |

**JSONL log entry** — written to `backend/logs/ingestion_YYYYMMDD.jsonl` after each run:

```json
{
  "timestamp": "2026-06-18T06:12:59+00:00",
  "step": "arxiv_fetch",
  "query": "(all categories)",
  "categories": ["cs.AI", "cs.LG", "cs.CL", "cs.CV"],
  "date_from": "2026-06-11T06:12:59+00:00",
  "date_to": "2026-06-18T06:12:59+00:00",
  "fetched": 10,
  "skipped_duplicates": 0,
  "latency_ms": 1569
}
```

---

### What this component intentionally does NOT do

| Excluded concern | Reason |
|---|---|
| Full PDF text extraction | Abstracts are sufficient for the "is this worth covering?" judgment. PDF parsing adds significant complexity and noise. |
| Filtering by relevance | Relevance judgment is the retrieval system's job, not the fetcher's. Ingest broad, filter later. |
| Error handling / retries | Out of scope for Day 1. The arXiv API is reliable enough for a dev setup. |
| Full-text search by query | That's the Day 2 `search_arxiv` agent tool — a deliberately separate concern. |

---

### Known limitations

- **`max_results=100` cap:** For a weekly fetch across 4 active ML categories, the real volume could be 1,000–3,000+ papers. The cap means we get the 100 most recent by submission date, not the full set. For the Day 1 prototype this is fine; a production setup would set `max_results` much higher or paginate without a cap.
- **No logging shared with other components yet:** The `_write_log()` helper writes JSONL directly. Once the shared `StepLogger` is built (Day 1 Step 7), this should be refactored to use it for consistency.
- **State file is not committed to git:** `backend/data/` is gitignored. If the state file is lost, the next run bootstraps fresh (re-ingests `bootstrap_days` of papers). The ID dedup in the vector store (built later) acts as a second safety net.

---

*Next: Component 2 — RSS Fetcher (`backend/ingestion/rss_fetcher.py`)*
