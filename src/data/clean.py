import pandas as pd

def build_text_column(
    frame: pd.DataFrame,
    text_column: str,
    title_column: str | None = None,
    combine_title_text: bool = False,
) -> pd.Series:
    """Build the document text column, optionally combining title and body."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")
    
    if text_column not in frame.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    text = frame[text_column].fillna("").astype(str).str.strip()

    if combine_title_text and title_column is not None:
        if title_column not in frame.columns:
            raise ValueError(f"Column '{title_column}' not found in DataFrame")
        
        title = frame[title_column].fillna("").astype(str).str.strip()
        combined = title.where(title == "", title + ". ") + text
        return combined.str.strip()

    return text


def drop_invalid_rows(
    frame: pd.DataFrame,
    text_column: str,
    label_column: str,
) -> pd.DataFrame:
    """Remove rows with empty text or missing labels."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")
    
    if text_column not in frame.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    if label_column not in frame.columns:
        raise ValueError(f"Column '{label_column}' not found in DataFrame")

    cleaned = frame.copy()
    cleaned = cleaned[cleaned[label_column].notna()]
    cleaned = cleaned[cleaned[text_column].astype(str).str.strip() != ""]

    dropped = len(frame) - len(cleaned)
    if dropped:
        print(f"Dropped {dropped} invalid dataset rows")

    return cleaned.reset_index(drop=True)
