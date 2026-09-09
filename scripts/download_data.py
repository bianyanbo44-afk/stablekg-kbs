"""Download and verify the public StableKG archives."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path


DATASETS = {
    "ICEWS14": {
        "url": "https://huggingface.co/datasets/linxy/ICEWS14/resolve/462141693b1c9c2ece49384fab41955eae89471a/zips/all.zip?download=true",
        "archive": "ICEWS14_all.zip",
        "sha256": "0BF107F5413221FEF3703F5025814015D8E329D2FE31CA8446F06561A0B17C77",
    },
    "ICEWS05-15": {
        "url": "https://huggingface.co/datasets/linxy/ICEWS05_15/resolve/17a1a7af550d7d61fa2b0a718dc742fcdc31fa11/zips/all.zip?download=true",
        "archive": "ICEWS05_15_all.zip",
        "sha256": "ADAE2A8D958BE85D04C1494F70CAF0F5D8F6C06CAF29BE518A7A3541FF4A3286",
    },
    "GDELT": {
        "url": "https://huggingface.co/datasets/linxy/GDELT/resolve/8ee57955b595b4edddf39b5bb93ea3309bc3233f/zips/all.zip?download=true",
        "archive": "GDELT_all.zip",
        "sha256": "34837172579C8577AA2CA78F29EC0DFC2DF0D798FD408188C829A0943C61242C",
    },
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest().upper()


def download_one(name: str, root: Path) -> None:
    spec = DATASETS[name]
    root.mkdir(parents=True, exist_ok=True)
    archive = root / spec["archive"]
    if not archive.exists():
        print(f"Downloading {name} ...", flush=True)
        with urllib.request.urlopen(spec["url"]) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
    actual = digest(archive)
    if actual != spec["sha256"]:
        raise RuntimeError(f"{name}: SHA-256 mismatch: expected {spec['sha256']}, got {actual}")
    target = root / archive.stem
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(target)
    required = [target / item for item in ("train.jsonl", "valid.jsonl", "test.jsonl")]
    if not all(item.exists() for item in required):
        raise RuntimeError(f"{name}: archive did not contain all required JSONL files")
    print(f"{name}: verified {actual}; extracted to {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        download_one(name, args.root)


if __name__ == "__main__":
    main()
