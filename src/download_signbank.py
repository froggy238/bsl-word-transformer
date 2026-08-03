"""Polite downloader for the BSL SignBank cross-corpus test set.

Downloads the citation-form MP4s listed in ``data/signbank_gloss_map.csv``
(built by hand/agent from the SignBank search UI) into
``data/test_videos_signbank/{word}/{word}_signbank_{nnn}.mp4`` — the
folder-per-word layout that ``src.extract`` mirrors and
``src.evaluate --split test`` consumes.

Run only under UCL's written permission for research use (see decision_log.md,
2026-07-27; permission email archived in the ethics appendix). Clips are used
solely as a held-out evaluation set, are never redistributed, and the
repository ships derived landmarks and this manifest, not video.

Politeness mirrors src.download: sequential requests, >= 1.5 s between
requests, an identifying User-Agent, idempotent skip of existing files.

Usage:
    python -m src.download_signbank [--manifest data/signbank_gloss_map.csv]
        [--max-per-word 3] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import time
import urllib.request
from datetime import date
from pathlib import Path

MANIFEST = "data/signbank_gloss_map.csv"
OUT_DIR = Path("data/test_videos_signbank")
PROVENANCE = Path("data/signbank_metadata.csv")
DELAY_S = 1.5
USER_AGENT = (
    "bsl-word-transformer/1.0 (MSc research; UCL BSL SignBank written "
    "permission on file; contact: https://github.com/froggy238)"
)


def fetch(url: str, dest: Path) -> tuple[int, int]:
    """Download url -> dest atomically. Returns (http_status, bytes)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
        status = resp.status
    tmp.write_bytes(data)
    tmp.replace(dest)
    return status, len(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--max-per-word", type=int, default=3)
    parser.add_argument(
        "--dry-run", action="store_true", help="list planned downloads only"
    )
    args = parser.parse_args()

    # utf-8-sig transparently strips a BOM if the manifest was written by
    # PowerShell/Excel; plain UTF-8 files read identically.
    with open(args.manifest, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("video_url", "").strip()]

    per_word: dict[str, int] = {}
    planned: list[tuple[dict, Path]] = []
    for row in rows:
        word = row["word"].strip()
        n = per_word.get(word, 0)
        if n >= args.max_per_word:
            continue
        per_word[word] = n + 1
        dest = OUT_DIR / word / f"{word}_signbank_{n + 1:03d}.mp4"
        planned.append((row, dest))

    print(f"{len(planned)} clips over {len(per_word)} words planned")
    if args.dry_run:
        for row, dest in planned:
            print(f"  {dest}  <-  {row['video_url']}")
        return

    prov_exists = PROVENANCE.is_file()
    with open(PROVENANCE, "a", encoding="utf-8", newline="") as prov:
        writer = csv.writer(prov)
        if not prov_exists:
            writer.writerow(
                ["filename", "word", "gloss", "entry_url", "video_url",
                 "download_date", "http_status", "bytes"]
            )
        done = skipped = failed = 0
        for row, dest in planned:
            if dest.is_file():
                skipped += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                status, size = fetch(row["video_url"].strip(), dest)
            except Exception as exc:  # log and continue: one miss must not abort the set
                print(f"FAILED {dest.name}: {exc}")
                failed += 1
                time.sleep(DELAY_S)
                continue
            writer.writerow(
                [dest.name, row["word"], row.get("gloss", ""),
                 row.get("entry_url", ""), row["video_url"],
                 date.today().isoformat(), status, size]
            )
            done += 1
            print(f"{dest.relative_to(OUT_DIR.parent)}  ({size / 1024:.0f} KB)")
            time.sleep(DELAY_S)
    print(f"downloaded {done}, skipped {skipped} existing, failed {failed}")


if __name__ == "__main__":
    main()
