"""
tests/test_input_handler.py

Basic tests for core/input_handler.py (Milestone 2).

Run with:
    pytest tests/test_input_handler.py -v

These tests create small sample files on the fly (in a temp folder) so
anyone on the team can run them without needing pre-made sample files.
"""

import os
import tempfile
import pytest

import fitz
from PIL import Image, ImageDraw

from core.input_handler import extract_text, InputExtractionError


def test_plain_text_passthrough():
    """Plain typed text should be returned unchanged (just stripped)."""
    result = extract_text("  Hello world  ")
    assert result == "Hello world"


def test_pdf_extraction(tmp_path):
    """Text inside a real PDF should be correctly extracted."""
    pdf_path = os.path.join(tmp_path, "sample.pdf")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello from a test PDF")
    doc.save(pdf_path)
    doc.close()

    result = extract_text(pdf_path)
    assert "Hello from a test PDF" in result


def test_image_ocr_extraction(tmp_path):
    """Text drawn onto an image should be readable via OCR."""
    image_path = os.path.join(tmp_path, "sample.png")
    image = Image.new("RGB", (400, 100), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 30), "TEST OCR TEXT", fill="black")
    image.save(image_path)

    result = extract_text(image_path)
    # OCR isn't always perfect, so we check case-insensitively and
    # just look for a recognizable chunk of the text.
    assert "TEST" in result.upper()


def test_unsupported_file_type_raises_error(tmp_path):
    """An unrecognized file extension should raise a friendly error."""
    bad_file_path = os.path.join(tmp_path, "sample.xyz")
    with open(bad_file_path, "w") as f:
        f.write("irrelevant content")

    with pytest.raises(InputExtractionError):
        extract_text(bad_file_path)
