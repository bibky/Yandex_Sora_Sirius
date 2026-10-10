
"""Parse application logs and extract severity statistics."""

import json
import re
from collections import Counter
from pathlib import Path


LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")

LOG_PATTERN = re.compile(
    r"\b(CRITICAL|ERROR|WARNING|WARN|INFO|DEBUG)\b",
    re.IGNORECASE,
)


def parse_logs(file_path: str | Path) -> dict:
    """Read a log file and summarize messages by severity."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Log file not found: {path}")

    counts = Counter({level: 0 for level in LOG_LEVELS})
    errors = []
    warnings = []
    total_lines = 0

    with path.open("r", encoding="utf-8") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            total_lines += 1
            match = LOG_PATTERN.search(line)

            if not match:
                continue

            level = match.group(1).upper()

            if level == "WARN":
                level = "WARNING"

            counts[level] += 1

            entry = {
                "line": line_number,
                "level": level,
                "message": line.strip(),
            }

            if level in ("ERROR", "CRITICAL"):
                errors.append(entry)
            elif level == "WARNING":
                warnings.append(entry)

    return {
        "source_file": path.name,
        "total_lines": total_lines,
        "counts": dict(counts),
        "errors": errors,
        "warnings": warnings,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse application logs."
    )
    parser.add_argument("file_path", help="Path to the log file")
    args = parser.parse_args()

    result = parse_logs(args.file_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
