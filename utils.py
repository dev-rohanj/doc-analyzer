"""
utils.py
--------
Shared helper utilities used across the project.
"""

import re


def clean_text(text: str) -> str:
    """
    Clean raw extracted PDF text:
      - Remove excessive whitespace
      - Remove non-printable characters
      - Normalize line breaks

    Args:
        text: Raw text from PDF extractor.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Replace multiple spaces/tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)

    # Replace more than 2 consecutive newlines with 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove non-printable characters (keep standard ASCII + unicode letters)
    text = re.sub(r'[^\x20-\x7E\n\u00C0-\u024F]', '', text)

    return text.strip()


def word_count(text: str) -> int:
    """Return approximate word count of a text string."""
    return len(text.split()) if text else 0


def truncate_text(text: str, max_chars: int = 500) -> str:
    """
    Truncate text to a maximum character limit, appending '...' if cut.

    Args:
        text:      Input string.
        max_chars: Maximum characters allowed.

    Returns:
        Truncated string.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def format_percentage(value: int) -> str:
    """Format an integer as a percentage string, e.g. 75 → '75%'."""
    return f"{value}%"


def severity_emoji(severity: str) -> str:
    """Return an emoji for a given risk severity level."""
    return {
        "high":   "🔴",
        "medium": "🟡",
        "low":    "🟢",
    }.get(severity.lower(), "⚪")


def clause_icon(clause_name: str) -> str:
    """Return an emoji icon for a given clause type."""
    icons = {
        "Termination Clause":                    "🔚",
        "Payment Clause":                        "💰",
        "Liability Clause":                      "⚖️",
        "Confidentiality Clause":                "🔒",
        "Indemnity Clause":                      "🛡️",
        "Dispute Resolution Clause":             "🏛️",
        "Obligations Clause":                    "📋",
        "Warranty Clause":                       "✅",
        "Intellectual Property Clause":          "💡",
        "Force Majeure Clause":                  "🌪️",
        "Governing Law Clause":                  "📜",
        "Non-Compete / Non-Solicitation Clause": "🚫",
    }
    return icons.get(clause_name, "📄")
