import re


def extract_entities(text: str):

    entities = {
        "amount": [],
        "transaction_id": [],
        "payment_method": [],
        "merchant": []
    }

    # Amounts: ₹500, ₹25,000, $500
    amount_pattern = r'(?:₹|\$|€|£)\s?\d+(?:,\d{3})*(?:\.\d{2})?'

    entities["amount"] = re.findall(
        amount_pattern,
        text,
        flags=re.IGNORECASE
    )

    # Transaction IDs
    transaction_pattern = r'\b(?:TXN|TRANSACTION)[-_]?[A-Z0-9]+\b'

    entities["transaction_id"] = re.findall(
        transaction_pattern,
        text,
        flags=re.IGNORECASE
    )

    # Payment methods
    payment_methods = [
        "UPI",
        "credit card",
        "debit card",
        "card",
        "ATM",
        "NEFT",
        "RTGS",
        "IMPS"
    ]

    text_lower = text.lower()

    for method in payment_methods:
        if method.lower() in text_lower:
            entities["payment_method"].append(method)

    return entities