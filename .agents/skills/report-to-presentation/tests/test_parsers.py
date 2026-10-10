
"""Unit tests for log and report parsers."""

import sys
from pathlib import Path

import pytest


SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from parse_logs import parse_logs
from parse_reports import parse_reports


def test_parse_logs_counts(tmp_path):
    """Verify that log severity levels are counted correctly."""
    log_file = tmp_path / "test.log"
    log_file.write_text(
        "INFO Application started\n"
        "WARNING High memory usage\n"
        "ERROR Connection failed\n"
        "CRITICAL Service unavailable\n",
        encoding="utf-8",
    )

    result = parse_logs(log_file)

    assert result["total_lines"] == 4
    assert result["counts"]["INFO"] == 1
    assert result["counts"]["WARNING"] == 1
    assert result["counts"]["ERROR"] == 1
    assert result["counts"]["CRITICAL"] == 1
    assert len(result["errors"]) == 2


def test_parse_logs_warn_alias(tmp_path):
    """Verify that WARN is normalized to WARNING."""
    log_file = tmp_path / "test.log"
    log_file.write_text("WARN Disk usage high\n", encoding="utf-8")

    result = parse_logs(log_file)

    assert result["counts"]["WARNING"] == 1


def test_parse_empty_log(tmp_path):
    """Verify that an empty log produces zero counts."""
    log_file = tmp_path / "empty.log"
    log_file.write_text("", encoding="utf-8")

    result = parse_logs(log_file)

    assert result["total_lines"] == 0
    assert sum(result["counts"].values()) == 0
    assert result["errors"] == []
    assert result["warnings"] == []


def test_parse_logs_missing_file(tmp_path):
    """Verify that a missing log raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        parse_logs(tmp_path / "missing.log")


def test_parse_reports_sections(tmp_path):
    """Verify that Markdown sections are extracted correctly."""
    report_file = tmp_path / "report.md"
    report_file.write_text(
        "# Weekly Report\n"
        "## Summary\n"
        "Application is stable.\n"
        "## Issues\n"
        "Database timeout detected.\n",
        encoding="utf-8",
    )

    result = parse_reports(report_file)

    assert result["section_count"] == 2
    assert result["sections"][0]["title"] == "Summary"
    assert result["sections"][1]["title"] == "Issues"
    assert "Database timeout" in result["sections"][1]["content"]


def test_parse_reports_invalid_extension(tmp_path):
    """Verify that unsupported report formats are rejected."""
    report_file = tmp_path / "report.csv"
    report_file.write_text("test", encoding="utf-8")

    with pytest.raises(ValueError):
        parse_reports(report_file)


def test_parse_reports_missing_file(tmp_path):
    """Verify that a missing report raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        parse_reports(tmp_path / "missing.md")
