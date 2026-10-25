"""
Tests for preprocessing and input validation logic in TruthLens AI.
"""

import pytest
from preprocessing import validate_input_text, clean_text, MAX_CHAR_LENGTH


def test_validate_input_valid_text():
    is_valid, msg = validate_input_text("Breaking news: Scientists discover new planetary system.")
    assert is_valid is True
    assert msg == ""


def test_validate_input_empty_and_whitespace():
    is_valid, msg = validate_input_text("")
    assert is_valid is False
    assert "empty" in msg.lower() or "whitespace" in msg.lower()

    is_valid, msg = validate_input_text("   \n\t  ")
    assert is_valid is False
    assert "whitespace" in msg.lower()


def test_validate_input_none():
    is_valid, msg = validate_input_text(None)
    assert is_valid is False
    assert "empty" in msg.lower() or "none" in msg.lower()


def test_validate_input_non_string():
    is_valid, msg = validate_input_text(12345)
    assert is_valid is False
    assert "expected string" in msg.lower()


def test_validate_input_no_alphanumeric():
    is_valid, msg = validate_input_text("!@#$%^&*()_+=-{}[]:;")
    assert is_valid is False
    assert "alphanumeric" in msg.lower()


def test_validate_input_exceeds_max_length():
    huge_text = "a" * (MAX_CHAR_LENGTH + 10)
    is_valid, msg = validate_input_text(huge_text)
    assert is_valid is False
    assert "exceeds maximum allowed length" in msg.lower()


def test_clean_text_url_removal():
    raw = "Check this link https://example.com/breaking-news and www.news-site.org today."
    cleaned = clean_text(raw)
    assert "https" not in cleaned
    assert "example.com" not in cleaned
    assert "www" not in cleaned
    assert "check this link" in cleaned
    assert "today" in cleaned


def test_clean_text_html_tag_removal():
    raw = "<p>President delivered a speech <b>today</b> in Washington.</p>"
    cleaned = clean_text(raw)
    assert "<p>" not in cleaned
    assert "<b>" not in cleaned
    assert "president delivered a speech today in washington" in cleaned


def test_clean_text_special_characters():
    raw = "ALERT: #Huge scandal uncovered in D.C.!!!"
    cleaned = clean_text(raw)
    assert "#" not in cleaned
    assert "!" not in cleaned
    assert ":" not in cleaned
    assert "alert huge scandal uncovered in" in cleaned
    assert "scandal" in cleaned
