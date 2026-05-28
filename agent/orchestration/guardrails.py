"""Rule-based input and output guardrails.

These guardrails are the always-on, provider-agnostic safety layer. They redact
common personally identifiable information and block configured topics and
oversized outputs. When the Bedrock backend is configured with a Guardrail
identifier, that managed guardrail runs server-side in addition to this layer;
the two are complementary, not exclusive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Order matters: the specific financial patterns must run before the generic
# phone pattern, which would otherwise greedily consume their digit runs.
_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)"),
}
_REDACTION = "[REDACTED:{label}]"


@dataclass(slots=True)
class GuardrailResult:
    """The outcome of a guardrail check.

    Attributes
    ----------
    allowed : bool
        Whether the text may proceed. ``False`` means the agent must block or
        escalate.
    text : str
        The text after redaction. Equal to the input when nothing is redacted.
    violations : list of str
        Machine-readable violation labels, for example ``"blocked_topic"`` or
        ``"redacted:email"``.
    """

    allowed: bool
    text: str
    violations: list[str] = field(default_factory=list)


class Guardrails:
    """Input and output guardrails for the agent.

    Parameters
    ----------
    blocked_topics : list of str or None, optional
        Lowercased substrings that, if present, block the text.
    max_output_chars : int, optional
        Maximum allowed length of an output. Longer outputs are blocked so the
        agent can escalate rather than emit a runaway response.
    """

    def __init__(
        self,
        blocked_topics: list[str] | None = None,
        max_output_chars: int = 6000,
    ) -> None:
        self.blocked_topics = [topic.lower() for topic in (blocked_topics or [])]
        self.max_output_chars = max_output_chars

    def check_input(self, text: str) -> GuardrailResult:
        """Validate and redact a user input.

        Parameters
        ----------
        text : str
            The raw user input.

        Returns
        -------
        GuardrailResult
            The check result. Blocked-topic hits set ``allowed=False``; PII is
            redacted but does not block input.
        """
        violations: list[str] = []

        topic_hit = self._blocked_topic(text)
        if topic_hit is not None:
            violations.append(f"blocked_topic:{topic_hit}")
            return GuardrailResult(allowed=False, text=text, violations=violations)

        redacted, redaction_labels = self._redact(text)
        violations.extend(redaction_labels)
        return GuardrailResult(allowed=True, text=redacted, violations=violations)

    def check_output(self, text: str) -> GuardrailResult:
        """Validate and redact a model output.

        Parameters
        ----------
        text : str
            The candidate output.

        Returns
        -------
        GuardrailResult
            The check result. Blocked topics or oversized output set
            ``allowed=False``; PII is redacted.
        """
        violations: list[str] = []

        topic_hit = self._blocked_topic(text)
        if topic_hit is not None:
            violations.append(f"blocked_topic:{topic_hit}")
            return GuardrailResult(allowed=False, text=text, violations=violations)

        if len(text) > self.max_output_chars:
            violations.append("output_too_long")
            return GuardrailResult(allowed=False, text=text, violations=violations)

        redacted, redaction_labels = self._redact(text)
        violations.extend(redaction_labels)
        return GuardrailResult(allowed=True, text=redacted, violations=violations)

    def _blocked_topic(self, text: str) -> str | None:
        """Return the first blocked topic found in ``text``, if any.

        Parameters
        ----------
        text : str
            Text to scan.

        Returns
        -------
        str or None
            The matched topic, or ``None`` if none match.
        """
        lowered = text.lower()
        for topic in self.blocked_topics:
            if topic and topic in lowered:
                return topic
        return None

    @staticmethod
    def _redact(text: str) -> tuple[str, list[str]]:
        """Redact known PII patterns from ``text``.

        Parameters
        ----------
        text : str
            Text to redact.

        Returns
        -------
        tuple of (str, list of str)
            The redacted text and the labels of redactions applied.
        """
        redacted = text
        labels: list[str] = []
        for label, pattern in _PII_PATTERNS.items():
            redacted, count = pattern.subn(
                _REDACTION.format(label=label), redacted
            )
            if count:
                labels.append(f"redacted:{label}")
        return redacted, labels
