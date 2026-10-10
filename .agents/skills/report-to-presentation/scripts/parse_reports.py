
"""Parse text and Markdown reports into structured sections."""

import argparse
import json
import re
from pathlib import Path


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def parse_reports(file_path: str | Path) -> dict:
    """Extract report sections from a text or Markdown file."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Report file not found: {path}")

    if path.suffix.lower() not in (".txt", ".md"):
        raise ValueError("Supported report formats are .txt and .md")

    content = path.read_text(encoding="utf-8")

    sections = []
    current_section = {
        "title": "Introduction",
        "level": 0,
        "content": [],
    }

    for line in content.splitlines():
        match = HEADING_PATTERN.match(line)

        if match:
            if current_section["content"]:
                sections.append(current_section)

            current_section = {
                "title": match.group(2).strip(),
                "level": len(match.group(1)),
                "content": [],
            }
        elif line.strip():
            current_section["content"].append(line.strip())

    if current_section["content"]:
        sections.append(current_section)

    return {
        "source_file": path.name,
        "section_count": len(sections),
        "sections": [
            {
                "title": section["title"],
                "level": section["level"],
                "content": "\n".join(section["content"]),
            }
            for section in sections
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse text and Markdown reports."
    )
    parser.add_argument("file_path", help="Path to the report file")
    args = parser.parse_args()

    result = parse_reports(args.file_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
