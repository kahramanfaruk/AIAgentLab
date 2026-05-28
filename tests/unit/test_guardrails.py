"""Tests for the rule-based guardrails."""

from __future__ import annotations

from agent.orchestration.guardrails import Guardrails


def test_input_redacts_email_but_allows() -> None:
    guardrails = Guardrails()
    result = guardrails.check_input("Contact me at john.doe@example.com please.")
    assert result.allowed is True
    assert "john.doe@example.com" not in result.text
    assert "redacted:email" in result.violations


def test_blocked_topic_blocks_input() -> None:
    guardrails = Guardrails(blocked_topics=["self-harm"])
    result = guardrails.check_input("I need help with self-harm.")
    assert result.allowed is False
    assert any(v.startswith("blocked_topic") for v in result.violations)


def test_output_too_long_is_blocked() -> None:
    guardrails = Guardrails(max_output_chars=10)
    result = guardrails.check_output("x" * 50)
    assert result.allowed is False
    assert "output_too_long" in result.violations


def test_output_redacts_iban() -> None:
    guardrails = Guardrails()
    result = guardrails.check_output("Pay to DE89370400440532013000 now.")
    assert result.allowed is True
    assert "DE89370400440532013000" not in result.text
    assert "redacted:iban" in result.violations


def test_clean_text_passes_unchanged() -> None:
    guardrails = Guardrails()
    text = "The deductible is 250 EUR."
    result = guardrails.check_output(text)
    assert result.allowed is True
    assert result.text == text
    assert result.violations == []
