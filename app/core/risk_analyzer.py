"""Pydantic models and validation logic for LLM-generated risk analysis output."""

import logging

from pydantic import BaseModel, field_validator, model_validator

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = {"red_flags", "yellow_flags", "green_flags"}


class RiskFlag(BaseModel):
    """A single annotated clause from a risk analysis.

    Attributes:
        clause: Name or description of the contract clause.
        explanation: Plain-language explanation of what the clause means.
        advice: Recommended action or consideration for the reader.
    """

    clause: str
    explanation: str
    advice: str

    @field_validator("clause", "explanation", "advice", mode="before")
    @classmethod
    def must_be_non_empty(cls, v: object) -> str:
        """Ensure each field is a non-empty string.

        Args:
            v: Raw field value from the LLM response.

        Returns:
            The validated string value.

        Raises:
            ValueError: If the value is not a non-empty string.
        """
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Field must be a non-empty string")
        return v.strip()


class RiskReport(BaseModel):
    """Complete risk analysis report with three severity categories.

    Attributes:
        red_flags: Clauses that are unfair, risky, or heavily one-sided.
        yellow_flags: Clauses that are missing, vague, or need clarification.
        green_flags: Clauses that are fair and protect the reader's interests.
    """

    red_flags: list[RiskFlag] = []
    yellow_flags: list[RiskFlag] = []
    green_flags: list[RiskFlag] = []

    @model_validator(mode="after")
    def warn_if_empty(self) -> "RiskReport":
        """Log a warning when all flag categories are empty.

        Returns:
            The validated RiskReport instance.
        """
        if not self.red_flags and not self.yellow_flags and not self.green_flags:
            logger.warning("Risk analysis returned no flags in any category")
        return self


def parse_risk_response(raw: dict) -> RiskReport:
    """Validate and coerce a raw LLM risk-analysis dict into a RiskReport.

    Accepts dicts that contain at least one of the three required keys.
    Malformed individual flag objects are skipped with a warning rather than
    raising an exception so that a partially valid response is still usable.

    Args:
        raw: Raw dict from the LLM, expected to have red_flags, yellow_flags,
            and/or green_flags keys.

    Returns:
        A validated RiskReport.

    Raises:
        ValueError: If none of the required top-level keys are present.
    """
    if not any(k in raw for k in _REQUIRED_KEYS):
        raise ValueError(
            f"Risk response missing all required keys {_REQUIRED_KEYS}. Got: {list(raw.keys())}"
        )

    validated: dict[str, list[RiskFlag]] = {
        "red_flags": [],
        "yellow_flags": [],
        "green_flags": [],
    }

    for key in _REQUIRED_KEYS:
        items = raw.get(key, [])
        if not isinstance(items, list):
            logger.warning("Key '%s' is not a list; skipping", key)
            continue
        for item in items:
            if not isinstance(item, dict):
                logger.warning("Skipping malformed flag (not a dict): %r", item)
                continue
            try:
                validated[key].append(RiskFlag(**item))
            except Exception as exc:
                logger.warning("Skipping malformed flag object — %s: %r", exc, item)

    return RiskReport(**validated)
