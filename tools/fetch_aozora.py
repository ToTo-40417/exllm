#!/usr/bin/env python3
"""Fetch rights-expired Aozora Bunko text files from the official index."""

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import re
import time
import unicodedata
import urllib.request
import zipfile
from pathlib import Path


USER_AGENT = "EXLLM-corpus-builder/1.0 (+https://github.com/ToTo-40417/EXLLM)"
SEPARATOR = re.compile(r"^-{20,}\s*$", re.MULTILINE)
AOZORA_NOTE = re.compile(r"［＃.*?］")
RUBY_BASE = re.compile(r"｜([^\n《]+)《[^\n》]*》")
RUBY = re.compile(r"《[^\n》]*》")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def decode_member(raw, declared):
    encodings = []
    if "UTF" in declared.upper():
        encodings.extend(("utf-8-sig", "utf-8"))
    encodings.extend(("cp932", "shift_jis", "utf-8-sig"))
    for encoding in encodings:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("aozora", raw, 0, min(1, len(raw)), "unsupported encoding")


def clean_text(text, title):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    separators = list(SEPARATOR.finditer(text))
    if len(separators) >= 2:
        text = text[separators[1].end() :]
    elif separators:
        text = text[separators[0].end() :]
    for marker in ("\n底本：", "\n底本:", "\n入力：", "\n※入力者"):
        position = text.find(marker)
        if position >= 0:
            text = text[:position]
    text = RUBY_BASE.sub(r"\1", text)
    text = RUBY.sub("", text)
    text = AOZORA_NOTE.sub("", text)
    text = text.replace("｜", "").replace("️", "")
    text = unicodedata.normalize("NFC", text)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t　]+", " ", line).strip()
        if line and not line.startswith("【テキスト中に現れる記号"):
            lines.append(line)
    body = "\n".join(lines).strip()
    if title and not body.startswith(title):
        body = title + "。\n" + body
    return body


def load_works(index_zip, orthography):
    with zipfile.ZipFile(index_zip) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError("expected exactly one CSV in index archive")
        text = archive.read(members[0]).decode("utf-8-sig")
    grouped = {}
    for row in csv.DictReader(io.StringIO(text)):
        grouped.setdefault(row["作品ID"], []).append(row)
    works = []
    for work_id, rows in grouped.items():
        first = rows[0]
        if first["作品著作権フラグ"] != "なし":
            continue
        if any(row["人物著作権フラグ"] != "なし" for row in rows):
            continue
        if orthography and first["文字遣い種別"] != orthography:
            continue
        url = first["テキストファイルURL"]
        if not url.startswith("https://www.aozora.gr.jp/") or not url.lower().endswith(".zip"):
            continue
        people = []
        for row in rows:
            person = {
                "id": row["人物ID"],
                "name": row["姓"] + row["名"],
                "role": row["役割フラグ"],
                "copyright": row["人物著作権フラグ"],
            }
            if person not in people:
                people.append(person)
        works.append({
            "work_id": work_id,
            "title": first["作品名"],
            "orthography": first["文字遣い種別"],
            "work_copyright": first["作品著作権フラグ"],
            "card_url": first["図書カードURL"],
            "text_url": url,
            "declared_encoding": first["テキストファイル符号化方式"],
            "text_updated": first["テキストファイル最終更新日"],
            "people": people,
        })
    return sorted(works, key=lambda item: item["work_id"])


def fetch_one(work, archive_dir, retries, delay):
    target = archive_dir / f"{work['work_id']}.zip"
    if target.exists():
        payload = target.read_bytes()
    else:
        request = urllib.request.Request(work["text_url"], headers={"User-Agent": USER_AGENT})
        error = None
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = response.read()
                temporary = target.with_suffix(".zip.part")
                temporary.write_bytes(payload)
                temporary.replace(target)
                if delay:
                    time.sleep(delay)
                break
            except Exception as exc:
                error = exc
                time.sleep(1 + attempt * 2)
        else:
            raise error
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".txt") and not name.startswith("__MACOSX/")]
        if not members:
            raise ValueError("no text member")
        member = max(members, key=lambda name: archive.getinfo(name).file_size)
        raw = archive.read(member)
    decoded, actual_encoding = decode_member(raw, work["declared_encoding"])
    text = clean_text(decoded, work["title"])
    if len(text) < 100:
        raise ValueError("cleaned text is too short")
    metadata = dict(work)
    metadata.update({
        "archive_sha256": sha256_bytes(payload),
        "member": member,
        "member_sha256": sha256_bytes(raw),
        "decoded_as": actual_encoding,
        "characters": len(text),
    })
    return {"text": text, "source": metadata}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--orthography", default="新字新仮名")
    parser.add_argument("--max-works", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--delay", type=float, default=0.3, help="polite delay after each network download per worker")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = args.output_dir / "archives"
    archive_dir.mkdir(exist_ok=True)
    corpus_path = args.output_dir / "documents.jsonl"
    errors_path = args.output_dir / "errors.jsonl"
    works = load_works(args.index, args.orthography)
    if args.max_works:
        works = works[: args.max_works]
    stats = {"eligible": len(works), "written": 0, "errors": 0, "characters": 0, "orthography": args.orthography}
    with corpus_path.open("w", encoding="utf-8") as corpus, errors_path.open("w", encoding="utf-8") as errors:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(fetch_one, work, archive_dir, args.retries, args.delay) for work in works]
            for work, future in zip(works, futures):
                try:
                    record = future.result()
                    corpus.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats["written"] += 1
                    stats["characters"] += record["source"]["characters"]
                except Exception as exc:
                    errors.write(json.dumps({"work_id": work["work_id"], "url": work["text_url"], "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False) + "\n")
                    stats["errors"] += 1
                if (stats["written"] + stats["errors"]) % 100 == 0:
                    print(json.dumps(stats, ensure_ascii=False), flush=True)
    stats["documents_sha256"] = sha256_file(corpus_path)
    (args.output_dir / "fetch-manifest.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
