import calendar
import feedparser
import json
import random
import requests
import time
import trafilatura
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

FEEDS = {
    "Hugging Face Blog": "https://huggingface.co/blog/feed.xml",
    "The Gradient":      "https://thegradient.pub/rss/",
    "Google DeepMind":   "https://deepmind.google/blog/rss.xml",
    "VentureBeat AI":    "https://venturebeat.com/category/ai/feed/",
}

# HuggingFace exposes its full post history (800+ entries). Other feeds may
# also grow over time. Cap at the most recent N entries per feed to keep
# weekly runs bounded (~4 feeds × 20 entries × 1s avg delay ≈ 80 seconds).
MAX_ENTRIES_PER_FEED = 20

RATE_LIMIT_MIN = 0.5  # seconds
RATE_LIMIT_MAX = 1.5

STATE_FILE = Path(__file__).parent.parent / "data" / "rss_state.json"
LOG_DIR    = Path(__file__).parent.parent / "logs"


@dataclass
class Article:
    title:       str
    url:         str
    published:   datetime | None
    source:      str
    text:        str
    text_source: str  # "full_text" | "rss_summary"


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {"last_fetched_at": None, "seen_urls": []}
    return json.loads(STATE_FILE.read_text())


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _write_log(entry: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"ingestion_{datetime.now().strftime('%Y%m%d')}.jsonl"
    with log_file.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _parse_published(entry) -> datetime | None:
    # feedparser normalises published_parsed to UTC as a time.struct_time.
    # calendar.timegm converts UTC struct_time → Unix timestamp (no TZ shift).
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        return datetime.fromtimestamp(
            calendar.timegm(entry.published_parsed), tz=timezone.utc
        )
    return None


FETCH_TIMEOUT = 10  # seconds — prevents hanging on slow or unresponsive URLs

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; Riset-bot/1.0; +https://github.com/abelitovisese)"
    )
}


def _fetch_full_text(url: str) -> str | None:
    """Fetch and extract the main article body. Returns None on failure."""
    try:
        resp = requests.get(url, timeout=FETCH_TIMEOUT, headers=HEADERS)
        resp.raise_for_status()
    except Exception:
        return None
    text = trafilatura.extract(resp.text)
    # Guard against empty or near-empty extractions (e.g. paywalled pages)
    if not text or len(text.strip()) < 100:
        return None
    return text


def fetch_articles(feeds: dict[str, str] = None) -> list[Article]:
    """
    Fetch new articles from all configured RSS feeds.

    RSS has no date-range API, so we always fetch the full feed and rely on
    URL-based deduplication to skip already-seen articles. A random delay of
    0.5–1.5s is applied between article body fetches to be polite to servers.

    Falls back to the RSS summary field if full-text extraction fails.
    """
    if feeds is None:
        feeds = FEEDS

    state = _load_state()
    seen_urls: set[str] = set(state["seen_urls"])
    now = datetime.now(timezone.utc)
    t_start = time.monotonic()

    new_articles: list[Article] = []
    new_urls:     list[str]     = []
    skipped   = 0
    fallbacks = 0

    for source_name, feed_url in feeds.items():
        feed = feedparser.parse(feed_url)

        if feed.bozo and len(feed.entries) == 0:
            print(f"[rss] warning: could not parse feed '{source_name}' ({feed_url})")
            continue

        for entry in feed.entries[:MAX_ENTRIES_PER_FEED]:
            url = entry.get("link", "")
            if not url or url in seen_urls:
                skipped += 1
                continue

            time.sleep(random.uniform(RATE_LIMIT_MIN, RATE_LIMIT_MAX))

            text = _fetch_full_text(url)
            text_source = "full_text"

            if text is None:
                rss_summary = entry.get("summary", "").strip()
                if not rss_summary:
                    continue  # nothing usable — skip entirely
                text = rss_summary
                text_source = "rss_summary"
                fallbacks += 1

            new_articles.append(Article(
                title=entry.get("title", ""),
                url=url,
                published=_parse_published(entry),
                source=source_name,
                text=text,
                text_source=text_source,
            ))
            new_urls.append(url)

    latency_ms = int((time.monotonic() - t_start) * 1000)

    state["last_fetched_at"] = now.isoformat()
    state["seen_urls"] = list(seen_urls | set(new_urls))
    _save_state(state)

    _write_log({
        "timestamp": now.isoformat(),
        "step": "rss_fetch",
        "feeds": list(feeds.keys()),
        "fetched": len(new_articles),
        "skipped_duplicates": skipped,
        "fallbacks_to_rss_summary": fallbacks,
        "latency_ms": latency_ms,
    })

    print(
        f"[rss] fetched {len(new_articles)} new articles, "
        f"skipped {skipped} duplicates, {fallbacks} fallbacks ({latency_ms}ms)"
    )
    return new_articles


if __name__ == "__main__":
    print("Fetching articles from all RSS feeds...\n")
    articles = fetch_articles()
    for a in articles:
        print(f"  [{a.source}] via {a.text_source}")
        print(f"  Title:   {a.title}")
        print(f"  URL:     {a.url}")
        print(f"  Preview: {a.text[:200].strip()}")
        print()
    print(f"Run again to confirm deduplication — all {len(articles)} URLs now in state.")
