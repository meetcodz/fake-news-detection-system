import pandas as pd
from pathlib import Path

def load_dataset(
    path: str | Path,
    text_column: str = "text",
    label_column: str = "label",
    title_column: str | None = None,
    combine_title_text: bool = False,
    drop_missing_text: bool = True,
    label_mapping: dict[Any, int] | None = None,
) -> pd.DataFrame:
    """Load a dataset from a CSV file and preprocess it."""
    if not isinstance(path, (str, Path)):
        raise TypeError("Expected a string or Path object for 'path'")
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    frame = pd.read_csv(path)

    if combine_title_text and title_column is not None:
        frame[text_column] = build_text_column(frame, text_column, title_column, combine_title_text)

    if drop_missing_text:
        frame = drop_invalid_rows(frame, text_column, label_column)

    if label_mapping:
        frame[label_column] = frame[label_column].map(label_mapping)

    return frame
