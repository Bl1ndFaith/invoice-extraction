"""LayoutLMv3 model factory for token classification."""

from __future__ import annotations

from transformers import AutoModelForTokenClassification, AutoProcessor

BASE_MODEL = "microsoft/layoutlmv3-base"


def load_processor(base_model: str = BASE_MODEL) -> AutoProcessor:
    """Processor with our own OCR: apply_ocr=False (we provide words+boxes)."""

    return AutoProcessor.from_pretrained(base_model, apply_ocr=False)


def load_model(num_labels: int, base_model: str = BASE_MODEL):
    return AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=num_labels,
    )
