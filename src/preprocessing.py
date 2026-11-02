import re
from bs4 import BeautifulSoup


def clean_text(text: str) -> str:
    """
    Clean customer support ticket text.
    """

    if not text:
        return ""

    # Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text(" ")

    # Mask email addresses
    text = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
        '[EMAIL]',
        text
    )

    # Mask phone numbers
    text = re.sub(
        r'\b(?:\+91[-\s]?)?[6-9]\d{9}\b',
        '[PHONE]',
        text
    )

    # Mask long account/card-like numbers
    text = re.sub(
        r'\b\d{10,16}\b',
        '[ACCOUNT_NUMBER]',
        text
    )

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()