"""GraphQL client for bina.az apartment sale listings.

bina.az migrated to a Next.js + Apollo GraphQL stack (discovered 2026-09).
Listings are fetched via ``POST https://bina.az/graphql`` with cursor-based
pagination — no authentication required.

The client is deliberately conservative:

* A realistic browser ``User-Agent`` header and ``Origin``/``Referer`` are set
  so the request looks like a normal Apollo client call.
* ``PAGE_DELAY`` (default 1.5 s) throttles between pages.
* ``page_size`` is capped at 50 (the server's observed maximum).

The public entry point is :func:`fetch_listings`, which yields one dict per
listing node with flattened fields ready for normalisation.
"""

from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)

GRAPHQL_URL = "https://bina.az/graphql"
PAGE_SIZE = 25   # bina.az GraphQL caps at 25 regardless of request
PAGE_DELAY = 1.0  # seconds between paginated requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/138.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Origin": "https://bina.az",
    "Referer": "https://bina.az/alqi-satqi/menziller",
}

ITEMS_QUERY = """
query FetchListings($first: Int!, $after: String) {
    itemsConnection(
        filter: {categoryId: "1", leased: false}
        first: $first
        after: $after
    ) {
        totalCount
        pageInfo { hasNextPage endCursor }
        edges {
            node {
                id
                path
                rooms
                floor
                floors
                area { value }
                price { total currency }
                location { id name latitude longitude }
                city { id name }
                photos { thumbnail large }
                hasRepair
                hasMortgage
                isFeatured
                updatedAt
            }
        }
    }
}
"""


def _flatten_node(node: dict) -> dict:
    """Flatten a GraphQL ESItem node into a simple dict for normalisation."""
    price = node.get("price") or {}
    area = node.get("area") or {}
    location = node.get("location") or {}
    city = node.get("city") or {}
    photos = node.get("photos") or []

    return {
        "listing_id": node.get("id"),
        "path": node.get("path"),
        "rooms": node.get("rooms"),
        "floor": node.get("floor"),
        "floors": node.get("floors"),
        "area": area.get("value"),
        "price": price.get("total"),
        "currency": price.get("currency"),
        "location_name": location.get("name"),
        "lat": location.get("latitude"),
        "lon": location.get("longitude"),
        "city": city.get("name"),
        "photos": [p.get("large") or p.get("thumbnail") for p in photos],
        "has_repair": node.get("hasRepair"),
        "has_mortgage": node.get("hasMortgage"),
        "is_featured": node.get("isFeatured"),
        "updated_at": node.get("updatedAt"),
    }


def fetch_listings(
    max_pages: int = 5,
    page_size: int = PAGE_SIZE,
    session: requests.Session | None = None,
) -> list[dict]:
    """Fetch apartment sale listings from bina.az GraphQL API.

    Returns a list of flattened dicts, one per listing. Fetches up to
    ``max_pages`` pages of ``page_size`` items each (max 50 per page).
    """
    page_size = min(page_size, 50)
    sess = session or requests.Session()
    sess.headers.update(HEADERS)

    listings: list[dict] = []
    cursor: str | None = None

    for page in range(1, max_pages + 1):
        variables: dict = {"first": page_size}
        if cursor:
            variables["after"] = cursor

        try:
            resp = sess.post(
                GRAPHQL_URL,
                json={"query": ITEMS_QUERY, "variables": variables},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Page %d failed: %s", page, exc)
            break

        if "errors" in data:
            logger.warning("GraphQL errors on page %d: %s", page, data["errors"])
            break

        connection = data.get("data", {}).get("itemsConnection", {})
        edges = connection.get("edges") or []

        if page == 1:
            total = connection.get("totalCount", "?")
            logger.info("bina.az reports %s total sale listings", total)

        for edge in edges:
            node = edge.get("node")
            if node:
                listings.append(_flatten_node(node))

        page_info = connection.get("pageInfo", {})
        cursor = page_info.get("endCursor")
        has_next = page_info.get("hasNextPage", False)

        logger.info(
            "Page %d: fetched %d listings (total so far: %d)",
            page, len(edges), len(listings),
        )

        if not has_next or not cursor:
            break
        if page < max_pages:
            time.sleep(PAGE_DELAY)

    return listings
