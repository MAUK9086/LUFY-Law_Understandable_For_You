"""Tests for risk-analysis response validation."""

import pytest

from app.core.risk_analyzer import parse_risk_response


def _flag(name: str) -> dict:
    return {"clause": name, "explanation": "what it means", "advice": "do this"}


def test_parse_valid_response():
    raw = {
        "red_flags": [_flag("Auto-renewal")],
        "yellow_flags": [_flag("Vague notice period")],
        "green_flags": [_flag("Clear rent amount")],
    }
    report = parse_risk_response(raw)
    assert len(report.red_flags) == 1
    assert report.red_flags[0].clause == "Auto-renewal"
    assert len(report.yellow_flags) == 1
    assert len(report.green_flags) == 1


def test_parse_skips_malformed_flags_but_keeps_valid():
    raw = {
        "red_flags": [
            _flag("Good flag"),
            {"clause": "Missing fields"},  # missing explanation/advice
            "not a dict",
            {"clause": "", "explanation": "x", "advice": "y"},  # empty clause
        ],
    }
    report = parse_risk_response(raw)
    assert len(report.red_flags) == 1
    assert report.red_flags[0].clause == "Good flag"


def test_parse_missing_all_keys_raises():
    with pytest.raises(ValueError):
        parse_risk_response({"something_else": []})


def test_parse_non_list_value_skipped():
    raw = {"red_flags": "oops", "green_flags": [_flag("Fair clause")]}
    report = parse_risk_response(raw)
    assert report.red_flags == []
    assert len(report.green_flags) == 1
