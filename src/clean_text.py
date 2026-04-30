"""
src/clean_text.py
------------------
Handles all raw text cleaning:
  - lowercase conversion
  - URL removal
  - mention/hashtag removal
  - punctuation & special character removal
  - extra whitespace removal
"""

import re
import string


def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def remove_urls(text: str) -> str:
    """Remove http/https URLs and www links."""
    return re.sub(r"http\S+|www\.\S+", "", text)


def remove_mentions(text: str) -> str:
    """Remove @username mentions."""
    return re.sub(r"@\w+", "", text)


def remove_hashtags(text: str) -> str:
    """Remove #hashtag symbols (keep the word)."""
    return re.sub(r"#", "", text)


def remove_punctuation(text: str) -> str:
    """Remove all punctuation characters."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text: str) -> str:
    """Remove standalone numbers."""
    return re.sub(r"\b\d+\b", "", text)


def remove_extra_spaces(text: str) -> str:
    """Collapse multiple spaces into one and strip."""
    return re.sub(r"\s+", " ", text).strip()


def remove_emojis(text: str) -> str:
    """Remove emoji characters using unicode range."""
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline applied in correct order.

    Args:
        text: raw social media text string

    Returns:
        cleaned text string
    """
    if not isinstance(text, str):
        return ""
    text = to_lowercase(text)
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_hashtags(text)
    text = remove_emojis(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_extra_spaces(text)
    return text


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "Zomato delivery was super fast today! Loved the packaging 🎉 #Zomato @zomato_support",
        "Netflix keeps buffering 😡 http://netflixhelp.com Check this!",
        "Received my Amazon package. Will open it later. 123",
    ]
    print("=" * 60)
    print("TEXT CLEANING MODULE – TEST OUTPUT")
    print("=" * 60)
    for s in samples:
        print(f"\nOriginal : {s}")
        print(f"Cleaned  : {clean_text(s)}")
