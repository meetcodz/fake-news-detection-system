"""
TruthLens — Hugging Face Spaces Deployment Script
Run once to push the project to your HF Space.

Usage:
    python scripts/deploy_to_hf.py --username YOUR_HF_USERNAME --space YOUR_SPACE_NAME --token YOUR_HF_TOKEN

Get your token from: https://huggingface.co/settings/tokens (needs write access)
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

INCLUDE_DIRS = [
    "app",
    "src",
    "utils",
    "configs",
    "data",
    "frontend",
    "models",
]

INCLUDE_FILES = [
    "Dockerfile",
    "requirements-api.txt",
    "pyproject.toml",
    "README_HF.md",
]

EXCLUDE_MODEL_DIRS = [
    "models/transformer/distilbert-base-uncased/checkpoints",
]

def run(cmd: list[str], cwd: str = None, check: bool = True) -> int:
    print(f"  > {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=check)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy TruthLens to Hugging Face Spaces")
    parser.add_argument("--username", required=True, help="Your Hugging Face username")
    parser.add_argument("--space", required=True, help="Space name (e.g. truthlens)")
    parser.add_argument("--token", required=True, help="HF write token from huggingface.co/settings/tokens")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.resolve()
    space_id = f"{args.username}/{args.space}"
    repo_url = f"https://user:{args.token}@huggingface.co/spaces/{space_id}"

    print(f"\n{'='*60}")
    print(f"Deploying TruthLens to HF Space: {space_id}")
    print(f"{'='*60}\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        deploy_dir = Path(tmpdir) / "deploy"
        deploy_dir.mkdir()

        print("[1/5] Cloning HF Space repo (or initializing)...")
        clone_result = run(
            ["git", "clone", repo_url, str(deploy_dir)],
            check=False
        )
        if clone_result != 0:
            print("  Space not found — will initialize fresh repo")
            run(["git", "init"], cwd=str(deploy_dir))
            run(["git", "remote", "add", "origin", repo_url], cwd=str(deploy_dir))

        print("[2/5] Copying project files...")
        for dir_name in INCLUDE_DIRS:
            src = project_root / dir_name
            if not src.exists():
                print(f"  Skipping {dir_name}/ (not found)")
                continue
            dst = deploy_dir / dir_name
            if dst.exists():
                shutil.rmtree(dst)
            exclude_paths = [project_root / ex for ex in EXCLUDE_MODEL_DIRS]
            shutil.copytree(src, dst, ignore=_build_ignore(exclude_paths))
            print(f"  Copied {dir_name}/")

        for file_name in INCLUDE_FILES:
            src = project_root / file_name
            if not src.exists():
                print(f"  Skipping {file_name} (not found)")
                continue
            shutil.copy2(src, deploy_dir / file_name)

        readme_src = deploy_dir / "README_HF.md"
        readme_dst = deploy_dir / "README.md"
        if readme_src.exists():
            shutil.copy2(readme_src, readme_dst)
            readme_src.unlink()
            print("  Renamed README_HF.md → README.md")

        gitignore_src = project_root / ".gitignore.hfspace"
        if gitignore_src.exists():
            shutil.copy2(gitignore_src, deploy_dir / ".gitignore")
            print("  Copied .gitignore (HF Space version)")

        lfs_extensions = ["*.pt", "*.joblib", "*.safetensors", "*.bin", "*.pkl"]
        print("[3/5] Configuring Git LFS for large model files...")
        run(["git", "lfs", "install"], cwd=str(deploy_dir), check=False)
        for ext in lfs_extensions:
            run(["git", "lfs", "track", ext], cwd=str(deploy_dir), check=False)

        run(["git", "config", "user.email", "deploy@truthlens.ai"], cwd=str(deploy_dir))
        run(["git", "config", "user.name", "TruthLens Deploy"], cwd=str(deploy_dir))

        print("[4/5] Staging and committing all files...")
        run(["git", "add", "--all"], cwd=str(deploy_dir))
        run(
            ["git", "commit", "-m", "deploy: TruthLens production release"],
            cwd=str(deploy_dir),
            check=False
        )

        print("[5/5] Pushing to Hugging Face Spaces...")
        run(["git", "push", "--force", "origin", "HEAD:main"], cwd=str(deploy_dir))

    print(f"\n{'='*60}")
    print("Deployment complete!")
    print(f"Your Space: https://huggingface.co/spaces/{space_id}")
    print(f"{'='*60}\n")
    print("Note: HF Spaces builds the Docker image automatically.")
    print("Check build progress at: https://huggingface.co/spaces/{space_id}/logs")


def _build_ignore(exclude_paths: list[Path]):
    def _ignore(src_str: str, names: list[str]) -> set[str]:
        src = Path(src_str)
        ignored = set()
        for name in names:
            full = src / name
            for ex in exclude_paths:
                try:
                    full.relative_to(ex)
                    ignored.add(name)
                    break
                except ValueError:
                    pass
                if full == ex:
                    ignored.add(name)
                    break
        return ignored
    return _ignore


if __name__ == "__main__":
    main()
