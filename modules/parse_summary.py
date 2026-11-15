"""Summary zone: tax and total (and the sub total, which is only used to validate the items)."""
import re

from modules.text_utils import find_numbers

INT_RE = re.compile(r"\d+(?: \d{3})*")


def _numbers(line):
    """Amounts with decimals; if there are none, whole numbers after the label ('Total Due : 2331')."""
    nums = find_numbers(line)
    if nums:
        return nums
    value_part = re.sub(r"^[^\d]*", "", line)
    return [int(m.replace(" ", "")) for m in INT_RE.findall(value_part)][-1:]


def parse_summary(text):
    tax = total = sub = None
    for line in text.splitlines():
        low = line.lower().strip()
        if re.match(r"^sub\s*total", low):
            nums = _numbers(line)
            sub = nums[-1] if nums else None
            continue
        if re.match(r"^(grand\s+)?total", low):
            nums = _numbers(line)
            if nums:
                total = nums[-1]              # gross amount is the last column
                if len(nums) >= 3:
                    tax = nums[-2]            # net worth | VAT | gross worth
        elif tax is None and re.match(r"^(vat|tax|sales tax)\b", low):
            nums = find_numbers(line)
            if len(nums) == 1:
                tax = nums[0]
    return {"tax": tax, "total": total, "sub": sub}
