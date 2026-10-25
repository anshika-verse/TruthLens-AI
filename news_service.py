"""
TruthLens AI — Multi-Source News Ingestion Service (news_service.py)
Fetches and normalizes authentic, real-time news from verified global publishers:
- BBC News
- The Guardian
- Google News Verified Aggregator (Reuters, AP, Bloomberg, etc.)
- NDTV News
Includes deduplication, image extraction, source telemetry, and local bookmark persistence.
"""

import os
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import html
import re
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 TruthLens/1.2"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BOOKMARKS_FILE = os.path.join(DATA_DIR, "bookmarks.json")

# Verified multi-source feed definitions
SOURCES = {
    "Google News (Verified Feeds)": {
        "All": "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
        "World": "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-US&gl=US&ceid=US:en",
        "India": "https://news.google.com/rss/headlines/section/topic/NATION?hl=en-IN&gl=IN&ceid=IN:en",
        "Politics": "https://news.google.com/rss/search?q=politics&hl=en-US&gl=US&ceid=US:en",
        "Technology": "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-US&gl=US&ceid=US:en",
        "Business": "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en",
        "Science": "https://news.google.com/rss/headlines/section/topic/SCIENCE?hl=en-US&gl=US&ceid=US:en",
        "Health": "https://news.google.com/rss/headlines/section/topic/HEALTH?hl=en-US&gl=US&ceid=US:en",
        "Sports": "https://news.google.com/rss/headlines/section/topic/SPORTS?hl=en-US&gl=US&ceid=US:en",
    },
    "BBC News": {
        "All": "http://feeds.bbci.co.uk/news/rss.xml",
        "World": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "Technology": "http://feeds.bbci.co.uk/news/technology/rss.xml",
        "Business": "http://feeds.bbci.co.uk/news/business/rss.xml",
        "Science": "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
        "Health": "http://feeds.bbci.co.uk/news/health/rss.xml",
    },
    "The Guardian": {
        "All": "https://www.theguardian.com/world/rss",
        "World": "https://www.theguardian.com/world/rss",
        "Politics": "https://www.theguardian.com/politics/rss",
        "Technology": "https://www.theguardian.com/technology/rss",
        "Business": "https://www.theguardian.com/business/rss",
        "Science": "https://www.theguardian.com/science/rss",
    },
    "NDTV News": {
        "All": "https://feeds.feedburner.com/ndtvnews-top-stories",
        "India": "https://feeds.feedburner.com/ndtvnews-india-news",
        "World": "https://feeds.feedburner.com/ndtvnews-world-news",
        "Technology": "https://feeds.feedburner.com/gadgets360-latest",
        "Sports": "https://feeds.feedburner.com/ndtvsports-latest",
    }
}


def clean_html_summary(raw_html: str) -> str:
    """Strips HTML markup and unescapes entities to produce clean, concise text."""
    if not raw_html:
        return ""
    clean = re.sub(r"<[^>]+>", " ", raw_html)
    clean = html.unescape(clean)
    clean = re.sub(r"\s+([.,!?;:])", r"\1", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def parse_pubdate_to_dt(pubdate_str: str) -> Optional[datetime]:
    """Parses standard RFC-822 date formats to datetime object for accurate chronological sorting."""
    if not pubdate_str:
        return None
    # Common format: 'Sat, 17 Oct 2026 08:30:00 GMT'
    try:
        clean_str = pubdate_str.strip()
        # Strip timezone string if present
        clean_str = re.sub(r"\s+[A-Za-z]+$", "", clean_str)
        # Parse standard formats
        for fmt in ("%a, %d %b %Y %H:%M:%S", "%d %b %Y %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return datetime.strptime(clean_str, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    except Exception:
        pass
    return None


def format_pubdate_display(dt: Optional[datetime], raw_str: str = "") -> str:
    """Formats datetime into human-readable display string."""
    if dt:
        return dt.strftime("%b %d, %Y • %I:%M %p")
    if raw_str:
        return raw_str[:22]
    return "Recent"


def extract_thumbnail_url(item: ET.Element) -> Optional[str]:
    """Extracts genuine thumbnail URL from RSS item using common media namespaces."""
    # 1. Check media:thumbnail
    for tag in (
        "{http://search.yahoo.com/mrss/}thumbnail",
        "{http://search.yahoo.com/mrss/}content"
    ):
        el = item.find(tag)
        if el is not None:
            url = el.attrib.get("url")
            if url and (url.startswith("http://") or url.startswith("https://")):
                return url

    # 2. Check enclosure
    enc = item.find("enclosure")
    if enc is not None:
        enc_type = enc.attrib.get("type", "")
        enc_url = enc.attrib.get("url", "")
        if "image" in enc_type and enc_url:
            return enc_url

    # 3. Check for <img> in description
    desc = item.findtext("description", default="")
    match = re.search(r'<img[^>]+src=["\'](https?://[^"\']+)["\']', desc)
    if match:
        return match.group(1)

    return None


def fetch_from_single_source(
    source_name: str,
    feed_url: str,
    category: str,
    timeout: int = 5
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Fetches articles from a single RSS feed endpoint with strict timeout and telemetry tracking.
    """
    articles = []
    telemetry = {
        "source": source_name,
        "status": "Operational",
        "last_updated": datetime.now().strftime("%I:%M %p"),
        "count": 0,
        "error": None
    }

    try:
        req = urllib.request.Request(feed_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()

        root = ET.fromstring(data)
        items = root.findall(".//item")

        for item in items:
            raw_title = item.findtext("title", default="").strip()
            link = item.findtext("link", default="").strip()
            pub_date_raw = item.findtext("pubDate", default="").strip()
            raw_desc = item.findtext("description", default="").strip()

            if not raw_title or not link:
                continue

            # Identify publisher
            source_tag = item.find("source")
            if source_tag is not None and source_tag.text:
                publisher = source_tag.text.strip()
            elif " - " in raw_title and source_name.startswith("Google"):
                parts = raw_title.rsplit(" - ", 1)
                publisher = parts[-1].strip()
                raw_title = parts[0].strip()
            else:
                publisher = source_name

            summary = clean_html_summary(raw_desc)
            if not summary or len(summary) < 20:
                summary = raw_title

            # Inshorts-style concise truncation (~70 words)
            words = summary.split()
            if len(words) > 70:
                concise_summary = " ".join(words[:70]) + "..."
            else:
                concise_summary = summary

            dt = parse_pubdate_to_dt(pub_date_raw)
            timestamp_disp = format_pubdate_display(dt, pub_date_raw)
            thumbnail = extract_thumbnail_url(item)

            # Generate stable deterministic ID
            item_id = hashlib.md5((link + raw_title).encode("utf-8")).hexdigest()[:12]

            articles.append({
                "id": item_id,
                "headline": raw_title,
                "summary": concise_summary,
                "source": publisher,
                "url": link,
                "image_url": thumbnail,
                "timestamp": timestamp_disp,
                "dt": dt,
                "category": category if category != "All" else "General",
                "retrieval_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        telemetry["count"] = len(articles)

    except Exception as e:
        telemetry["status"] = "Unavailable"
        telemetry["error"] = str(e)

    return articles, telemetry


def fetch_multi_source_news(
    category: str = "All",
    search_query: Optional[str] = None,
    enabled_sources: Optional[List[str]] = None,
    limit: int = 25
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Ingests news from all configured sources, deduplicates by canonical URL
    and normalized headline, and sorts chronologically.
    """
    all_articles = []
    source_telemetry = {}

    if enabled_sources is None:
        enabled_sources = list(SOURCES.keys())

    # 1. Search Query override uses Google News search
    if search_query and search_query.strip():
        q_enc = urllib.parse.quote_plus(search_query.strip())
        feed_url = f"https://news.google.com/rss/search?q={q_enc}&hl=en-US&gl=US&ceid=US:en"
        arts, telem = fetch_from_single_source("Search Feeds", feed_url, category)
        all_articles.extend(arts)
        source_telemetry["Search Feeds"] = telem
    else:
        # 2. Ingest from each configured source
        for src_name in enabled_sources:
            src_config = SOURCES.get(src_name)
            if not src_config:
                continue

            feed_url = src_config.get(category) or src_config.get("All")
            if not feed_url:
                continue

            arts, telem = fetch_from_single_source(src_name, feed_url, category)
            all_articles.extend(arts)
            source_telemetry[src_name] = telem

    # 3. Deduplication (by URL and normalized headline)
    seen_urls = set()
    seen_headlines = set()
    deduped = []

    for art in all_articles:
        url_key = art["url"].lower().rstrip("/")
        norm_title = re.sub(r"[^\w\s]", "", art["headline"].lower())[:45]

        if url_key in seen_urls or norm_title in seen_headlines:
            continue

        seen_urls.add(url_key)
        seen_headlines.add(norm_title)
        deduped.append(art)

    # 4. Sort chronologically (newest first where dt is available)
    deduped.sort(key=lambda x: x["dt"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    return deduped[:limit], source_telemetry


# Backward-compatible convenience alias
def fetch_live_news(category: str = "All", search_query: Optional[str] = None, limit: int = 15):
    arts, _ = fetch_multi_source_news(category=category, search_query=search_query, limit=limit)
    return arts


# -------------------------------------------------------------
# LIGHTWEIGHT LOCAL BOOKMARKS MANAGER
# -------------------------------------------------------------
class BookmarkManager:
    """Handles local bookmark persistence in data/bookmarks.json."""

    @staticmethod
    def _ensure_data_dir():
        os.makedirs(DATA_DIR, exist_ok=True)

    @classmethod
    def load_bookmarks(cls) -> Dict[str, Dict[str, Any]]:
        cls._ensure_data_dir()
        if os.path.exists(BOOKMARKS_FILE):
            try:
                with open(BOOKMARKS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    @classmethod
    def save_bookmark(cls, article: Dict[str, Any]) -> bool:
        cls._ensure_data_dir()
        bookmarks = cls.load_bookmarks()
        art_id = article.get("id")
        if not art_id:
            return False

        # Store minimal metadata only
        bookmarks[art_id] = {
            "id": art_id,
            "headline": article.get("headline", ""),
            "summary": article.get("summary", ""),
            "source": article.get("source", ""),
            "url": article.get("url", ""),
            "image_url": article.get("image_url"),
            "timestamp": article.get("timestamp", ""),
            "category": article.get("category", "General"),
            "saved_at": datetime.now().strftime("%b %d, %Y • %I:%M %p")
        }

        try:
            with open(BOOKMARKS_FILE, "w", encoding="utf-8") as f:
                json.dump(bookmarks, f, indent=2)
            return True
        except Exception:
            return False

    @classmethod
    def remove_bookmark(cls, article_id: str) -> bool:
        cls._ensure_data_dir()
        bookmarks = cls.load_bookmarks()
        if article_id in bookmarks:
            del bookmarks[article_id]
            try:
                with open(BOOKMARKS_FILE, "w", encoding="utf-8") as f:
                    json.dump(bookmarks, f, indent=2)
                return True
            except Exception:
                return False
        return False

    @classmethod
    def is_bookmarked(cls, article_id: str) -> bool:
        bookmarks = cls.load_bookmarks()
        return article_id in bookmarks
