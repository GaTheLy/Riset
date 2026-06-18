import arxiv
from dataclasses import dataclass
from datetime import datetime

CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV"]


@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    authors: list[str]
    published: datetime
    url: str
    categories: list[str]


def fetch_papers(
    query: str,
    categories: list[str] = None,
    max_results: int = 20,
) -> list[Paper]:
    if categories is None:
        categories = CATEGORIES

    # Build category filter, e.g. "cat:cs.AI OR cat:cs.LG OR ..."
    category_filter = " OR ".join(f"cat:{c}" for c in categories)
    full_query = f"({query}) AND ({category_filter})"

    client = arxiv.Client()
    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []
    for result in client.results(search):
        papers.append(Paper(
            id=result.entry_id,
            title=result.title,
            abstract=result.summary,
            authors=[a.name for a in result.authors],
            published=result.published,
            url=result.entry_id,
            categories=result.categories,
        ))

    return papers


if __name__ == "__main__":
    print("Fetching recent papers about 'large language models'...\n")
    papers = fetch_papers("large language models", max_results=5)

    for p in papers:
        print("=" * 60)
        print(f"Title:      {p.title}")
        authors_str = ", ".join(p.authors[:3])
        if len(p.authors) > 3:
            authors_str += f" (+{len(p.authors) - 3} more)"
        print(f"Authors:    {authors_str}")
        print(f"Published:  {p.published.strftime('%Y-%m-%d')}")
        print(f"Categories: {', '.join(p.categories)}")
        print(f"URL:        {p.url}")
        print(f"Abstract:   {p.abstract[:250].strip()}...")
        print()
