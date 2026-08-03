"""Strip all # comments and standalone docstrings from every Python file."""
import ast
import io
import os
import sys
import tokenize


SKIP_PATHS = {".venv", ".venv-stage2", "__pycache__", ".git", "strip_comments.py"}


def _collect_docstring_ranges(source: str) -> set[tuple[int, int]]:
    """Return (start_line, end_line) for every standalone docstring."""
    ranges: set[tuple[int, int]] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ranges

    for node in ast.walk(tree):
        body = None
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            ranges.add((first.lineno, first.end_lineno))
    return ranges


def strip_comments(source: str) -> str:
    docstring_lines: set[int] = set()
    for start, end in _collect_docstring_ranges(source):
        docstring_lines.update(range(start, end + 1))

    out_tokens = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError:
        return source

    skip_next_nl = False
    for tok in tokens:
        ttype, tstring, (srow, _scol), (_erow, _ecol), _ = tok

        if ttype == tokenize.COMMENT:
            skip_next_nl = True
            continue

        if ttype in (tokenize.NL, tokenize.NEWLINE) and skip_next_nl:
            skip_next_nl = False
            continue

        if ttype == tokenize.STRING and srow in docstring_lines:
            continue

        skip_next_nl = False
        out_tokens.append(tok)

    try:
        result = tokenize.untokenize(out_tokens)
    except Exception:
        return source

    lines = result.splitlines()
    clean: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        clean.append(line)
        prev_blank = is_blank

    return "\n".join(clean).strip() + "\n"


def process_file(path: str) -> None:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        original = fh.read()

    if not original.strip():
        return

    cleaned = strip_comments(original)

    if cleaned != original:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(cleaned)
        print(f"  cleaned  {path}")
    else:
        print(f"  no-op    {path}")


def main(root: str) -> None:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_PATHS and not d.startswith(".")
        ]
        for fname in filenames:
            if fname.endswith(".py") and fname not in SKIP_PATHS:
                process_file(os.path.join(dirpath, fname))


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"Stripping comments under: {os.path.abspath(root)}")
    main(root)
    print("Done.")
