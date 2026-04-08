"""
summarizer.py
-------------
Extractive text summarization using word-frequency scoring.
No AI APIs — pure Python NLP.

How it works:
  1. Tokenize text into sentences and words.
  2. Score each word by frequency (ignoring stopwords).
  3. Score each sentence as the sum of its word scores.
  4. Return the top N highest-scoring sentences as the summary.
"""

import re
from collections import Counter


# Common English stopwords to ignore during frequency scoring
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "that", "this", "these", "those", "it", "its", "they", "them", "their",
    "we", "our", "you", "your", "he", "she", "his", "her", "who", "which",
    "what", "when", "where", "how", "all", "each", "every", "both", "other",
    "such", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "then", "once",
    "not", "no", "nor", "so", "if", "as", "than", "too", "very", "just",
    "because", "while", "although", "however", "therefore", "also", "any",
    "more", "most", "own", "same", "only", "also", "up", "about",
}


def summarize(text: str, num_sentences: int = 7) -> str:
    """
    Generate an extractive summary of the document.

    Args:
        text:          Full document text.
        num_sentences: Number of sentences to include in the summary.

    Returns:
        A readable summary string.
    """
    # Step 1: Split text into sentences
    sentences = _split_sentences(text)

    if len(sentences) <= num_sentences:
        # Document is already short — return cleaned version
        return " ".join(sentences)

    # Step 2: Compute word frequencies across all sentences
    word_freq = _compute_word_frequencies(text)

    # Step 3: Score each sentence
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        words = _tokenize(sentence)
        score = sum(word_freq.get(w, 0) for w in words)
        # Normalize by sentence length to avoid bias toward long sentences
        if len(words) > 0:
            sentence_scores[i] = score / len(words)

    # Step 4: Pick top N sentences by score
    top_indices = sorted(
        sentence_scores,
        key=lambda i: sentence_scores[i],
        reverse=True
    )[:num_sentences]

    # Step 5: Return sentences in their ORIGINAL order (preserves readability)
    top_indices_sorted = sorted(top_indices)
    summary_sentences = [sentences[i] for i in top_indices_sorted]

    return " ".join(summary_sentences)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list:
    """
    Split text into sentences on '.', '!', '?', or newlines.
    Filters out very short or empty fragments.
    """
    text = text.replace("\n", " ").strip()
    # Split on sentence-ending punctuation followed by whitespace
    raw = re.split(r'(?<=[.!?])\s+', text)
    # Filter out fragments shorter than 30 characters
    return [s.strip() for s in raw if len(s.strip()) > 30]


def _tokenize(text: str) -> list:
    """
    Lowercase and split text into words, removing punctuation.
    """
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def _compute_word_frequencies(text: str) -> dict:
    """
    Count word frequencies across the entire document.
    Returns a dict of {word: normalized_frequency}.
    """
    words = _tokenize(text)
    counts = Counter(words)

    # Normalize: divide each count by the max count
    max_count = max(counts.values()) if counts else 1
    return {word: count / max_count for word, count in counts.items()}
