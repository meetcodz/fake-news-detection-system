import re

def preprocess_text(text: str, config: dict[str, Any] | None = None) -> str:
    """Clean and normalize a single text document."""
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")

    settings = config or {}
    
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", text)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = cleaned.lower()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    min_length = int(settings.get("min_text_length", 1))
    if len(cleaned) < min_length:
        raise ValueError(
            f"Text too short after preprocessing (min {min_length} chars)"
        )

    return cleaned


def preprocess_corpus(
    texts: list[str],
    config: dict[str, Any] | None = None,
) -> list[str]:
    """Preprocess a batch of documents, skipping invalid entries with logging."""
    processed, _ = preprocess_dataset(texts, labels=None, config=config)
    return processed


def preprocess_dataset(
    texts: list[str],
    labels: list[int] | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[list[str], list[int]] | tuple[list[str], None]:
    """Preprocess documents while keeping texts and labels aligned."""
    if labels is not None and len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")

    processed_texts: list[str] = []
    processed_labels: list[int] = []
    skipped = 0

    for index, text in enumerate(texts):
        try:
            processed_texts.append(preprocess_text(text, config))
            if labels is not None:
                processed_labels.append(labels[index])
        except ValueError as exc:
            skipped += 1
            print(f"Skipped document during preprocessing: {exc}")

    if not processed_texts:
        raise ValueError("No valid documents remain after preprocessing")

    if skipped:
        print(f"Preprocessing skipped {skipped} documents")

    if labels is None:
        return processed_texts, None
    return processed_texts, processed_labels
