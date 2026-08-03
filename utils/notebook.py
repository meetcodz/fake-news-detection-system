import sys
from pathlib import Path

def find_project_root(marker: str = "pyproject.toml") -> Path:
                                                                              
    candidate = Path.cwd().resolve()
    for path in (candidate, *candidate.parents):
        if (path / marker).exists():
            return path
    raise FileNotFoundError(
        f"Could not find project root containing '{marker}'. "
        "Open the notebook from the repository or a subdirectory."
    )

def setup_notebook_environment(
    install_package: bool = False,
    extras: str = "dev,notebook",
) -> Path:

    root = find_project_root()
    root_str = str(root)

    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    if install_package:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", f".[{extras}]"],
            cwd=root,
        )

    return root
