"""
TruthLens AI - Text Validation & Preprocessing Module
Provides consistent, leakage-safe text preprocessing for training and inference.
"""

import re
from typing import Tuple


MAX_CHAR_LENGTH = 50000  # Guard against memory exhaustion from oversized inputs


def validate_input_text(text: any) -> Tuple[bool, str]:
    """
    Validates whether the incoming input is non-empty, of string type,
    and contains sufficient readable text.

    Returns:
        (is_valid, error_or_warning_message)
    """
    if text is None:
        return False, "Input text cannot be empty or None."

    if not isinstance(text, str):
        return False, f"Expected string input, received {type(text).__name__}."

    stripped = text.strip()
    if not stripped:
        return False, "Input text contains only whitespace. Please enter headline or article content."

    if len(stripped) > MAX_CHAR_LENGTH:
        return False, f"Input text exceeds maximum allowed length ({MAX_CHAR_LENGTH} characters)."

    # Check if there are any alphanumeric characters
    if not re.search(r"[a-zA-Z0-9]", stripped):
        return False, "Input text must contain alphanumeric characters (words or numbers)."

    return True, ""


def clean_text(text: str) -> str:
    """
    Applies consistent text cleaning:
    - Casts to string and trims
    - Converts to lowercase
    - Strips URLs and web links
    - Strips HTML tags
    - Strips email addresses
    - Removes non-alphanumeric punctuation while preserving spaces
    - Normalizes consecutive whitespace
    """
    if not isinstance(text, str):
        text = str(text or "")

    # Lowercase
    cleaned = text.lower()

    # Strip HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    # Strip URLs
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

    # Strip emails
    cleaned = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", cleaned)

    # Remove special characters, brackets, and punctuation except single apostrophes in contractions
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)

    # Collapse multiple whitespaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned
