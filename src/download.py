"""Polite SignBSL.com scraper for word-level BSL clips.

For each vocabulary word, fetches https://www.signbsl.com/sign/{word},
collects the hosted mp4 variant URLs (with contributing organisation when
identifiable), and downloads them to data/raw_videos/{word}/{clip_id}.mp4,
appending rows to data/metadata.csv. Idempotent: source URLs already in the
metadata are skipped. All HTTP requests are separated by >= 1 second.

Downloaded videos are for local research use only and must never be
redistributed (data/raw_videos is gitignored).

CLI:
    python -m src.download [--dry-run] [--words hello,thank-you]
                           [--max-per-word 6]
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

SIGN_URL = "https://www.signbsl.com/sign/{word}"
USER_AGENT = (
    "BSL-MSc-Research/1.0 "
    "(academic research; contact: tarekonins@gmail.com)"
)
REQUEST_DELAY_S = 1.0
DEFAULT_MAX_PER_WORD = 6

METADATA_COLUMNS = [
    "word", "clip_id", "source", "organisation", "signer_id", "source_url",
    "video_file", "resolution", "duration_s", "fps", "download_date", "notes",
]

_MP4_RE = re.compile(r"https?://[^\s\"'<>\\]+\.mp4", re.IGNORECASE)

_last_request_time = 0.0


@dataclass
class Variant:
    """One downloadable video variant found on a sign page."""

    url: str
    organisation: str = "signbsl"


def _polite_get(
    session: requests.Session, url: str, **kwargs
) -> requests.Response:
    """GET with a module-wide >= REQUEST_DELAY_S gap between requests."""
    global _last_request_time
    wait = REQUEST_DELAY_S - (time.monotonic() - _last_request_time)
    if wait > 0:
        time.sleep(wait)
    try:
        return session.get(url, timeout=30, **kwargs)
    finally:
        _last_request_time = time.monotonic()


def _block_organisation(block) -> str | None:
    """Best-effort organisation name from a VideoObject-ish element."""
    for prop in ("publisher", "author"):
        holder = block.find(attrs={"itemprop": prop})
        if holder is None:
            continue
        name = holder.find(attrs={"itemprop": "name"})
        candidates = [name, holder] if name is not None else [holder]
        for tag in candidates:
            value = tag.get("content") or tag.get_text(strip=True)
            if value:
                return value
    return None


def _url_organisation(url: str) -> str | None:
    """Contributing organisation encoded in a media.signbsl.com URL path.

    Observed layouts: videos/bsl/{org}/mp4/file.mp4, videos/{org}/mp4/file.mp4
    and videos/bsl/{org}/file.mp4 — 'bsl' is a language folder and 'mp4' a
    format folder, neither is an organisation. This path segment is the most
    reliable per-clip source attribution on the site.
    """
    parsed = urlparse(url)
    if parsed.netloc.lower() != "media.signbsl.com":
        return None
    segments = [s for s in parsed.path.split("/") if s]
    if len(segments) < 3 or segments[0].lower() != "videos":
        return None
    middle = [s for s in segments[1:-1] if s.lower() != "mp4"]
    if middle and middle[0].lower() == "bsl":
        middle = middle[1:]
    return middle[0].lower() if middle else None


def parse_variants(html: str) -> list[Variant]:
    """Collect unique mp4 URLs (with organisation when identifiable).

    Liberal by design: scans schema.org VideoObject blocks, <video>/<source>
    tags, og:video metas and finally any mp4 URL in the raw markup.
    """
    soup = BeautifulSoup(html, "html.parser")
    found: dict[str, str] = {}  # url -> organisation, insertion-ordered

    def add(url: str | None, organisation: str | None = None) -> None:
        if not url or ".mp4" not in url.lower():
            return
        url = url.strip()
        org = organisation or _url_organisation(url)
        if url not in found:
            found[url] = org or "signbsl"
        elif org and found[url] == "signbsl":
            found[url] = org

    for block in soup.find_all(attrs={"itemtype": re.compile("VideoObject")}):
        org = _block_organisation(block)
        for meta in block.find_all(attrs={"itemprop": "contentUrl"}):
            add(meta.get("content") or meta.get("src"), org)
        for tag in block.find_all(["video", "source"]):
            add(tag.get("src") or tag.get("data-src"), org)

    for tag in soup.find_all(["video", "source"]):
        add(tag.get("src") or tag.get("data-src"))
    for meta in soup.find_all("meta"):
        if meta.get("property") in (
            "og:video", "og:video:url", "og:video:secure_url"
        ) or meta.get("itemprop") == "contentUrl":
            add(meta.get("content"))
    for link in soup.find_all("a", href=True):
        add(link["href"])
    for url in _MP4_RE.findall(html):
        add(url)

    return [Variant(url=url, organisation=org) for url, org in found.items()]


def fetch_variants(
    session: requests.Session, word: str
) -> list[Variant] | None:
    """Fetch and parse the sign page for ``word``; None if the page is 404."""
    url = SIGN_URL.format(word=word.strip().lower().replace(" ", "-"))
    resp = _polite_get(session, url)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return parse_variants(resp.text)


def _source_token(organisation: str) -> str:
    """Filename-safe lowercase source token for clip ids."""
    token = re.sub(r"[^a-z0-9]", "", organisation.lower())
    return token[:24] or "signbsl"


def load_metadata(path: str) -> list[dict[str, str]]:
    """Existing metadata rows as dicts (empty list if the file is absent)."""
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_metadata(path: str, rows: list[dict[str, str]]) -> None:
    """Atomically rewrite metadata.csv with the exact column order."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in METADATA_COLUMNS})
    os.replace(tmp, dest)


def _next_clip_index(rows: list[dict[str, str]], word: str, source: str) -> int:
    """Next 1-based index for clip ids of the form {word}_{source}_{nnn}."""
    prefix = f"{word}_{source}_"
    best = 0
    for row in rows:
        clip_id = row.get("clip_id", "")
        if clip_id.startswith(prefix):
            tail = clip_id[len(prefix):]
            if tail.isdigit():
                best = max(best, int(tail))
    return best + 1


def _probe_video(path: Path) -> tuple[str, str, str]:
    """(resolution, duration_s, fps) via OpenCV; blanks on failure."""
    try:
        import cv2
    except ImportError:
        return "", "", ""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return "", "", ""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    resolution = f"{width}x{height}" if width > 0 and height > 0 else ""
    fps_s = f"{fps:.2f}" if fps and fps > 0 else ""
    duration = f"{n_frames / fps:.2f}" if fps and fps > 0 and n_frames > 0 else ""
    return resolution, duration, fps_s


def download_variant(
    session: requests.Session, variant: Variant, dest: Path
) -> None:
    """Stream a variant to a temp file, then atomically rename into place."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    resp = _polite_get(session, variant.url, stream=True)
    resp.raise_for_status()
    try:
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, dest)
    finally:
        resp.close()
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _read_vocabulary_words(vocabulary_csv: str) -> list[str]:
    with open(vocabulary_csv, newline="", encoding="utf-8") as f:
        return [row["word"] for row in csv.DictReader(f) if row.get("word")]


def _process_word(
    session: requests.Session,
    word: str,
    rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> int:
    """Download new variants for one word; returns number downloaded."""
    variants = fetch_variants(session, word)
    if variants is None:
        print(f"SUBSTITUTION NEEDED: {word}")
        return 0
    if args.dry_run:
        print(f"{word}: {len(variants)} variants")
        return 0

    existing_urls = {row.get("source_url", "") for row in rows}
    n_for_word = sum(1 for row in rows if row.get("word") == word)
    downloaded = 0
    for variant in variants:
        if n_for_word >= args.max_per_word:
            break
        if variant.url in existing_urls:
            continue
        source = _source_token(variant.organisation)
        clip_id = f"{word}_{source}_{_next_clip_index(rows, word, source):03d}"
        dest = Path(args.out) / word / f"{clip_id}.mp4"
        try:
            download_variant(session, variant, dest)
        except requests.RequestException as exc:
            print(f"Warning: download failed for {variant.url}: {exc}")
            continue
        resolution, duration_s, fps = _probe_video(dest)
        rows.append({
            "word": word,
            "clip_id": clip_id,
            "source": source,
            "organisation": variant.organisation,
            "signer_id": "",
            "source_url": variant.url,
            "video_file": dest.as_posix(),
            "resolution": resolution,
            "duration_s": duration_s,
            "fps": fps,
            "download_date": date.today().isoformat(),
            "notes": "",
        })
        existing_urls.add(variant.url)
        write_metadata(args.metadata, rows)  # persist after every clip
        n_for_word += 1
        downloaded += 1
        print(f"Downloaded {clip_id} ({variant.organisation})")
    return downloaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Politely scrape SignBSL.com for vocabulary clips."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="only report per-word variant counts")
    parser.add_argument("--words", default=None,
                        help="comma-separated words (default: vocabulary.csv)")
    parser.add_argument("--max-per-word", type=int,
                        default=DEFAULT_MAX_PER_WORD,
                        help="cap on clips per word (default 6)")
    parser.add_argument("--vocabulary", default="data/vocabulary.csv")
    parser.add_argument("--out", default="data/raw_videos")
    parser.add_argument("--metadata", default="data/metadata.csv")
    args = parser.parse_args(argv)

    if args.words:
        words = [w.strip().lower().replace(" ", "-")
                 for w in args.words.split(",") if w.strip()]
    else:
        words = _read_vocabulary_words(args.vocabulary)

    rows = load_metadata(args.metadata)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    total = 0
    had_error = False
    for word in words:
        try:
            total += _process_word(session, word, rows, args)
        except requests.RequestException as exc:
            had_error = True
            print(f"Warning: request failed for '{word}': {exc}")
    if not args.dry_run:
        print(f"Done: {total} new clips; metadata rows: {len(rows)}")
    if args.dry_run:
        return 0  # dry runs always exit 0
    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
