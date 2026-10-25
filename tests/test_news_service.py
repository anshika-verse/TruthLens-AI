"""
Tests for news_service.py retrieval, parsing, multi-source ingestion, and bookmark manager.
"""

import pytest
from news_service import (
    clean_html_summary,
    parse_pubdate_to_dt,
    format_pubdate_display,
    fetch_live_news,
    fetch_multi_source_news,
    BookmarkManager
)


def test_clean_html_summary_stripping():
    html_input = "<p>Scientists at <b>NASA</b> announced a <a href='https://example.com'>major discovery</a>.</p>"
    cleaned = clean_html_summary(html_input)
    assert "<p>" not in cleaned
    assert "<b>" not in cleaned
    assert "<a" not in cleaned
    assert "Scientists at NASA announced a major discovery." in cleaned


def test_clean_html_summary_entities():
    html_input = "Markets &amp; inflation rose &gt; 2% in &quot;volatile&quot; trading."
    cleaned = clean_html_summary(html_input)
    assert "&amp;" not in cleaned
    assert "&quot;" not in cleaned
    assert 'Markets & inflation rose > 2% in "volatile" trading.' == cleaned


def test_parse_pubdate_to_dt_and_formatting():
    sample_date = "Sat, 17 Oct 2026 08:30:00 GMT"
    dt = parse_pubdate_to_dt(sample_date)
    formatted = format_pubdate_display(dt, sample_date)
    assert "Oct 17, 2026" in formatted


def test_fetch_multi_source_news_structure():
    articles, telemetry = fetch_multi_source_news("Technology", limit=4)
    assert isinstance(articles, list)
    assert isinstance(telemetry, dict)
    if articles:
        first = articles[0]
        assert "headline" in first
        assert "summary" in first
        assert "source" in first
        assert "url" in first
        assert "id" in first


def test_bookmark_lifecycle():
    sample_article = {
        "id": "test_bookmark_xyz123",
        "headline": "Breakthrough in Clean Energy Technology",
        "summary": "Researchers announced a scalable fusion battery.",
        "source": "Tech Journal",
        "url": "https://example.com/energy-news",
        "timestamp": "Today",
        "category": "Technology"
    }

    # Save
    saved = BookmarkManager.save_bookmark(sample_article)
    assert saved is True
    assert BookmarkManager.is_bookmarked("test_bookmark_xyz123") is True

    # Retrieve
    all_bms = BookmarkManager.load_bookmarks()
    assert "test_bookmark_xyz123" in all_bms
    assert all_bms["test_bookmark_xyz123"]["headline"] == "Breakthrough in Clean Energy Technology"

    # Remove
    removed = BookmarkManager.remove_bookmark("test_bookmark_xyz123")
    assert removed is True
    assert BookmarkManager.is_bookmarked("test_bookmark_xyz123") is False
