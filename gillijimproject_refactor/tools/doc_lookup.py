#!/usr/bin/env python3
"""
Local documentation lookup tool for gillijimproject_refactor.

Builds a compact searchable index across memory-bank, docs, and .github guidance,
then returns ranked matches with short summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")
HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_INDEX_PATH = PROJECT_ROOT / ".cache" / "doc_lookup_index.json"

DEFAULT_ROOTS = [
    PROJECT_ROOT / "memory-bank",
    PROJECT_ROOT / "src" / "MdxViewer" / "memory-bank",
    PROJECT_ROOT / "docs",
    WORKSPACE_ROOT / ".github",
]

SKIP_DIR_NAMES = {
    ".git",
    "bin",
    "obj",
    "node_modules",
    ".cache",
    "artifacts",
    "publish",
}

DEFAULT_EXTENSIONS = {
    ".md",
    ".txt",
    ".prompt.md",
    ".instructions.md",
    ".agent.md",
    ".skill.md",
}


def relative_to_workspace(path: Path) -> str:
    try:
        return path.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_summary(text: str, max_chars: int = 280) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    # Prefer first non-heading paragraph line.
    for line in lines:
        if not line.startswith("#"):
            return line[:max_chars]

    return lines[0][:max_chars]


def extract_title(path: Path, text: str) -> str:
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            return match.group(1).strip()
    return path.name


def iter_docs(roots: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix.lower() in DEFAULT_EXTENSIONS or root.name.lower().endswith(tuple(DEFAULT_EXTENSIONS)):
                resolved = root.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield resolved
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in SKIP_DIR_NAMES for part in path.parts):
                continue

            lower_name = path.name.lower()
            if path.suffix.lower() not in DEFAULT_EXTENSIONS and not lower_name.endswith(tuple(DEFAULT_EXTENSIONS)):
                continue

            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield resolved


def build_index(roots: List[Path], index_path: Path) -> None:
    docs = []
    doc_freq: Counter[str] = Counter()

    for file_path in iter_docs(roots):
        text = read_text(file_path)
        tokens = tokenize(text)
        if not tokens:
            continue

        counts = Counter(tokens)
        doc_freq.update(set(counts.keys()))

        docs.append(
            {
                "path": relative_to_workspace(file_path),
                "title": extract_title(file_path, text),
                "summary": extract_summary(text),
                "token_counts": dict(counts.most_common(512)),
                "token_total": len(tokens),
            }
        )

    index_data = {
        "workspace_root": str(WORKSPACE_ROOT),
        "doc_count": len(docs),
        "doc_freq": dict(doc_freq),
        "docs": docs,
    }

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index_data, indent=2), encoding="utf-8")

    print(f"Indexed {len(docs)} documents -> {index_path.as_posix()}")


def load_index(index_path: Path) -> Dict:
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found: {index_path.as_posix()}\n"
            f"Run: python gillijimproject_refactor/tools/doc_lookup.py build"
        )
    return json.loads(index_path.read_text(encoding="utf-8"))


def score_doc(query_tokens: List[str], doc: Dict, doc_freq: Dict[str, int], total_docs: int) -> float:
    token_counts: Dict[str, int] = doc.get("token_counts", {})
    if not token_counts:
        return 0.0

    score = 0.0
    title = doc.get("title", "").lower()
    path = doc.get("path", "").lower()

    for token in query_tokens:
        tf = token_counts.get(token, 0)
        if tf <= 0:
            continue

        df = max(1, int(doc_freq.get(token, 1)))
        idf = math.log((1 + total_docs) / df) + 1.0
        score += tf * idf

        if token in title:
            score += 2.0
        if token in path:
            score += 1.0

    return score


def query_index(index_path: Path, query_text: str, limit: int) -> None:
    data = load_index(index_path)
    docs = data.get("docs", [])
    total_docs = max(1, int(data.get("doc_count", len(docs))))
    doc_freq = data.get("doc_freq", {})

    query_tokens = tokenize(query_text)
    if not query_tokens:
        print("Query has no usable tokens.")
        return

    scored = []
    for doc in docs:
        score = score_doc(query_tokens, doc, doc_freq, total_docs)
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)

    if not scored:
        print("No matches found.")
        return

    print(f"Query: {query_text}")
    print(f"Matches: {min(limit, len(scored))}/{len(scored)}")
    print("-")

    for rank, (score, doc) in enumerate(scored[:limit], start=1):
        print(f"{rank}. [{doc['path']}] score={score:.2f}")
        print(f"   title: {doc.get('title', '')}")
        summary = doc.get("summary", "")
        if summary:
            print(f"   summary: {summary}")


def print_stats(index_path: Path) -> None:
    data = load_index(index_path)
    print(f"Index: {index_path.as_posix()}")
    print(f"Workspace: {data.get('workspace_root', '')}")
    print(f"Documents: {data.get('doc_count', 0)}")
    print(f"Unique tokens: {len(data.get('doc_freq', {}))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search project memory-bank/docs guidance quickly.")
    parser.add_argument("command", choices=["build", "query", "stats"], help="Action to run")
    parser.add_argument("query", nargs="?", default="", help="Query text for the query command")
    parser.add_argument("--index", default=str(DEFAULT_INDEX_PATH), help="Index JSON path")
    parser.add_argument("--limit", type=int, default=12, help="Max query results")
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Additional root folder or file to include (can be repeated)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_path = Path(args.index)

    if args.command == "build":
        roots = list(DEFAULT_ROOTS)
        if args.root:
            roots.extend(Path(root) for root in args.root)
        build_index(roots, index_path)
        return

    if args.command == "stats":
        print_stats(index_path)
        return

    if args.command == "query":
        if not args.query.strip():
            raise SystemExit("Provide query text. Example: query \"alpha decode profile\"")
        query_index(index_path, args.query, args.limit)
        return


if __name__ == "__main__":
    main()
