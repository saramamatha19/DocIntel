import re


# Emails have a fixed, recognizable shape, so a regex reliably
# catches them. Passwords don't have a fixed shape, but they
# almost always appear labeled ("password: ...", "pwd=...") in
# real documents, so matching the label + value is the practical
# way to catch them without a fixed pattern for the value itself.
#
# Names are NOT handled here on purpose: unlike emails or labeled
# passwords, names have no reliable structural signature ("John
# Smith" looks exactly like any two capitalized words), so regex
# genuinely cannot detect them reliably. That needs a trained NER
# (Named Entity Recognition) model, e.g. spaCy — a bigger, separate
# addition, not something to fake here with an unreliable heuristic.

EMAIL_PATTERN = re.compile(
    r"\b[\w.+-]+@[\w.-]+\.\w+\b"
)

PASSWORD_PATTERN = re.compile(
    r"(?i)\b(password|pwd|passwd)(\s*[:=]\s*)\S+"
)


def redact_pii(text: str) -> str:
    """
    Redact emails and labeled passwords from text before it's
    chunked, embedded, or sent to any LLM. The label stays
    visible (e.g. "password: [REDACTED_PASSWORD]") so it's clear
    a credential was present, without the actual value surviving.
    """

    text = EMAIL_PATTERN.sub(
        "[REDACTED_EMAIL]",
        text,
    )

    text = PASSWORD_PATTERN.sub(
        r"\1\2[REDACTED_PASSWORD]",
        text,
    )

    return text
