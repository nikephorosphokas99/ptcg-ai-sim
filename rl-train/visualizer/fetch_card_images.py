"""Download the card art and name it by the official engine's card ids.

    cd rl-train/visualizer
    python fetch_card_images.py          # needs git-lfs: conda install -c conda-forge git-lfs

The scans (~880 MB) live in Git LFS under ``simulator/card_images_en`` at an
older commit, named by the custom engine the simulator used to run on. The
official engine numbers some newer cards differently (Special Red Card is 1413
there, 1513 in the art), so ``card_image_map.json`` maps each official id to its
art, matched by card name, type and HP. This fetches those LFS objects into the
local git store and writes ``card_images_en/<official id>.webp``; the folder is
git-ignored rather than committed again.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "card_images_en"


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def main() -> None:
    spec = json.loads((HERE / "card_image_map.json").read_text())
    commit, source = spec["source_commit"], spec["source_dir"]
    repo = Path(git("rev-parse", "--show-toplevel", cwd=HERE).strip())
    git_dir = Path(git("rev-parse", "--absolute-git-dir", cwd=repo).strip())
    if shutil.which("git-lfs") is None:
        sys.exit("git-lfs not found: conda install -c conda-forge git-lfs (or your package manager)")

    print(f"fetching {source} at {commit} from Git LFS ...", flush=True)
    subprocess.run(["git", "lfs", "fetch", "origin", commit, f"--include={source}/*"], cwd=repo, check=True)

    OUT.mkdir(exist_ok=True)
    written = 0
    for official_id, art_id in spec["map"].items():
        pointer = git("show", f"{commit}:{source}/{art_id}.webp", cwd=repo)
        oid = next(line.split(":", 1)[1] for line in pointer.splitlines() if line.startswith("oid sha256:"))
        obj = git_dir / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
        if not obj.exists():
            sys.exit(f"LFS object for {art_id}.webp missing after fetch: {obj}")
        shutil.copyfile(obj, OUT / f"{official_id}.webp")
        written += 1
    print(f"wrote {written} images to {OUT}; no art for: {', '.join(spec['missing']) or 'none'}")


if __name__ == "__main__":
    main()
