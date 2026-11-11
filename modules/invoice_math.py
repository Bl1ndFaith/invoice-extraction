"""Arithmetic reconciliation: the error detector for extracted values.

Compares what the pipeline extracted, as money values:

    1. sum(items.total_price)  ~=  subtotal
    2. subtotal + tax          ~=  total
    3. sum(items.total_price)  ~=  total

The controller never fixes values — it only reports OK / WARN / SKIP per
check, so wrong extraction becomes visible without any template knowledge.
Tolerance: 0.5% of the larger side plus one cent (rounding, per-item cent
corrections).

All functions are pure: data in, data out. No printing, no file IO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

TOLERANCE_SHARE = 0.005
TOLERANCE_CENTS = 0.02


def money_value(text: str) -> float | None:
    """'4573' / '$0.00' / '232.95' / '15 339,50' / '6 416,12' -> float.

    Returns None when the string is not a parseable amount.
    """

    cleaned = text.replace("$", "").replace(" ", "").strip().rstrip(",.:;")
    if not cleaned:
        return None

    # Space-group rendering keeps the comma decimal ('15 339,50');
    # single values are already dot-decimals ('232.95').
    if cleaned.count(",") == 1 and cleaned.count(".") == 0:
        cleaned = cleaned.replace(",", ".")

    try:
        return float(cleaned)
    except ValueError:
        return None


def _tolerance(amount: float) -> float:
    return TOLERANCE_SHARE * abs(amount) + TOLERANCE_CENTS


@dataclass
class MathReport:
    """Check name -> human-readable status ('ok ...' / 'WARN ...' / 'skip ...')."""

    checks: dict[str, str]


def _fmt(amount: float) -> str:
    return f"{amount:.2f}"


def reconcile(
    item_totals: Sequence[str],
    subtotal: str,
    tax: str,
    total: str,
    shipping: str = "",
    discount: str = "",
) -> MathReport:
    """Run the three reconciliation checks on extracted amounts.

    shipping/discount are optional internal (non-schema) amounts; they are
    added into check 2 only when they were actually extracted.
    """

    items_sum: float | None = None
    parsed_items = [money_value(text) for text in item_totals]
    parsed_items = [value for value in parsed_items if value is not None]
    if parsed_items:
        items_sum = sum(parsed_items)

    subtotal_value = money_value(subtotal)
    tax_value = money_value(tax)
    total_value = money_value(total)

    checks: dict[str, str] = {}

    # 1. sum(items) ~= subtotal
    if items_sum is None or subtotal_value is None:
        checks["items_vs_subtotal"] = "skip (missing items sum or subtotal)"
    else:
        checks["items_vs_subtotal"] = _compare(items_sum, subtotal_value, "items")

    # 2. subtotal + tax + shipping - discount ~= total
    if subtotal_value is None or tax_value is None or total_value is None:
        checks["subtotal_tax_vs_total"] = "skip (missing subtotal/tax/total)"
    else:
        left = subtotal_value + tax_value
        adjustments = []
        shipping_value = money_value(shipping)
        if shipping_value is not None:
            left += shipping_value
            adjustments.append(f"+shipping {_fmt(shipping_value)}")
        discount_value = money_value(discount)
        if discount_value is not None:
            left -= discount_value
            adjustments.append(f"-discount {_fmt(discount_value)}")
        label = "subtotal+tax" + (" " + " ".join(adjustments) if adjustments else "")
        checks["subtotal_tax_vs_total"] = _compare(left, total_value, label)

    # 3. sum(items) ~= total — FALLBACK check, only when the subtotal was
    # not found at all (batch1-style templates); otherwise it duplicates
    # check 1 and adds noise (tax/shipping belong to the gap anyway).
    if items_sum is None or total_value is None:
        checks["items_vs_total"] = "skip (missing items sum or total)"
    elif subtotal_value is not None:
        checks["items_vs_total"] = "skip (subtotal present, check 1 covers it)"
    else:
        checks["items_vs_total"] = _compare(items_sum, total_value, "items")

    return MathReport(checks=checks)


def _compare(left: float, right: float, label: str) -> str:
    difference = abs(left - right)
    tolerance = _tolerance(max(abs(left), abs(right)))

    if difference <= tolerance:
        return f"ok: {label} {_fmt(left)} = {_fmt(right)} (diff {_fmt(difference)})"
    return (
        f"WARN: {label} {_fmt(left)} != {_fmt(right)} "
        f"(diff {_fmt(difference)} > tol {_fmt(tolerance)})"
    )
