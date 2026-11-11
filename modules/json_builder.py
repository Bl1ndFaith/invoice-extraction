"""JSON assembly and validation for the target output schema."""

from __future__ import annotations

import re

DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
DECIMAL_RE = re.compile(r"^-?\d+(\.\d+)?$")

INVOICE_KEYS = (
    "client_name",
    "client_address",
    "seller_name",
    "seller_address",
    "invoice_number",
    "invoice_date",
    "due_date",
)

ITEM_KEYS = ("description", "quantity", "total_price")
SUBTOTAL_KEYS = ("tax", "discount", "total")
PAYMENT_KEYS = ("due_date", "bank_name", "account_number", "payment_method")


def build_invoice_document(invoice: dict, items: list[dict], subtotal: dict) -> dict:
    """Assemble the final document with fixed key order and empty defaults."""

    return {
        "invoice": {key: str(invoice.get(key) or "") for key in INVOICE_KEYS},
        "items": [
            {key: str(item.get(key) or "") for key in ITEM_KEYS} for item in items
        ],
        "subtotal": {key: str(subtotal.get(key) or "") for key in SUBTOTAL_KEYS},
        "payment_instructions": {key: "" for key in PAYMENT_KEYS},
    }


def validate_document(document: dict) -> list[str]:
    """Return a list of schema violations (empty list means valid)."""

    errors: list[str] = []

    if set(document) != {"invoice", "items", "subtotal", "payment_instructions"}:
        errors.append("Top-level keys must be exactly the four schema sections.")

    invoice = document.get("invoice", {})
    if set(invoice) != set(INVOICE_KEYS):
        errors.append("invoice: missing or unexpected keys.")

    for key in INVOICE_KEYS:
        value = invoice.get(key)
        if not isinstance(value, str):
            errors.append(f"invoice.{key}: must be a string, got {type(value).__name__}.")

    if invoice.get("invoice_date") and not DATE_RE.match(invoice["invoice_date"]):
        errors.append(f"invoice.invoice_date: bad format {invoice['invoice_date']!r}.")

    for index, item in enumerate(document.get("items", [])):
        if set(item) != set(ITEM_KEYS):
            errors.append(f"items[{index}]: missing or unexpected keys.")
            continue
        for key in ITEM_KEYS:
            if not isinstance(item[key], str):
                errors.append(f"items[{index}].{key}: must be a string.")
        if item.get("total_price") and not DECIMAL_RE.match(item["total_price"]):
            errors.append(f"items[{index}].total_price: bad decimal format.")

    subtotal = document.get("subtotal", {})
    if set(subtotal) != set(SUBTOTAL_KEYS):
        errors.append("subtotal: missing or unexpected keys.")
    else:
        for key in SUBTOTAL_KEYS:
            if not isinstance(subtotal[key], str):
                errors.append(f"subtotal.{key}: must be a string.")
        if subtotal.get("total") and not DECIMAL_RE.match(subtotal["total"]):
            errors.append("subtotal.total: bad decimal format.")

    payment = document.get("payment_instructions", {})
    if set(payment) != set(PAYMENT_KEYS):
        errors.append("payment_instructions: missing or unexpected keys.")

    return errors
