from typing import Iterable, List

def normalize_texts(texts: Iterable[str]) -> List[str]:
    """Basic normalization: strip, lowercase, deduplicate."""
    seen = set()
    cleaned = []
    for t in texts:
        norm = t.strip().lower()
        if norm and norm not in seen:
            seen.add(norm)
            cleaned.append(norm)
    return cleaned
