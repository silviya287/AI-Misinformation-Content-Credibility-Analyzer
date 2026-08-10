"""
core/input_handler.py

Module A - Input Handling (Milestone 2).

This file has ONE job: no matter what the user gives us (typed text, a
PDF, an image, or a video), turn it into a plain Python string.
Everything after this file (preprocessing, ML) only ever deals with
plain text - it never needs to know where that text came from.

Public function every other module should use:
    extract_text(source: str) -> str

`source` can be:
    - a plain string the user typed (returned as-is)
    - a file path ending in .pdf / .png / .jpg / .jpeg / .bmp /
      .mp4 / .mov / .avi / .mkv

The four "private" helper functions below do the actual work for each
file type. They're still simple top-level functions (not classes) per
our finalized simplified architecture - no need for extractor classes
or interfaces here.
"""

import os
import tempfile

import fitz  # PyMuPDF, for reading PDFs
import pytesseract
from PIL import Image


# File extensions we recognize for each input type.
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


class InputExtractionError(Exception):
    """
    Raised whenever we can't turn the given input into text - e.g. a
    corrupt PDF, an unreadable image, or an unsupported file type.
    The UI layer will catch this and show a friendly error message
    instead of crashing.
    """
    pass


def extract_text(source: str) -> str:
    """
    Main entry point for Module A.

    Looks at `source` and decides:
      - if it's an existing file path, check its extension and route
        it to the right extractor (PDF / image / video)
      - otherwise, treat it as plain text typed by the user and return
        it unchanged

    Returns the extracted plain text as a string.
    Raises InputExtractionError if extraction fails.
    """

    # If it's not a path that exists on disk, we assume the user typed
    # or pasted plain text directly into the Analyze screen.
    if not os.path.isfile(source):
        return source.strip()

    # It IS a real file - figure out what kind by its extension.
    _, extension = os.path.splitext(source)
    extension = extension.lower()

    if extension in PDF_EXTENSIONS:
        return _extract_from_pdf(source)

    if extension in IMAGE_EXTENSIONS:
        return _extract_from_image(source)

    if extension in VIDEO_EXTENSIONS:
        return _extract_from_video(source)

    # If we get here, it's a file type we don't know how to handle yet.
    raise InputExtractionError(
        f"Unsupported file type: '{extension}'. "
        f"Supported types are PDF, image (png/jpg/bmp/tiff), "
        f"and video (mp4/mov/avi/mkv)."
    )


def _extract_from_pdf(file_path: str) -> str:
    """
    Extracts all text from a PDF file using PyMuPDF (imported as fitz).
    Loops through every page and joins the text together.
    """
    try:
        pdf_document = fitz.open(file_path)
    except Exception as error:
        raise InputExtractionError(f"Could not open PDF file: {error}")

    extracted_pages = []
    for page in pdf_document:
        extracted_pages.append(page.get_text())
    pdf_document.close()

    full_text = "\n".join(extracted_pages).strip()

    if not full_text:
        raise InputExtractionError(
            "No text could be found in this PDF. It may be a scanned "
            "PDF made only of images - try converting it to an image "
            "first, or use the image (OCR) option instead."
        )

    return full_text


def _extract_from_image(file_path: str) -> str:
    """
    Extracts text from an image using Tesseract OCR (via pytesseract).
    """
    try:
        image = Image.open(file_path)
        extracted_text = pytesseract.image_to_string(image)
    except Exception as error:
        raise InputExtractionError(f"Could not read text from image: {error}")

    extracted_text = extracted_text.strip()

    if not extracted_text:
        raise InputExtractionError(
            "No readable text was found in this image. Try a clearer "
            "or higher-resolution image."
        )

    return extracted_text


def _extract_from_video(file_path: str) -> str:
    """
    Extracts text from a video in two steps:
      1. Pull the audio track out of the video file (moviepy + ffmpeg).
      2. Run offline speech-to-text on that audio using Vosk.

    Vosk requires a downloaded language model folder on disk (this is
    what keeps the app fully offline - no cloud speech API calls).
    We look for it at VOSK_MODEL_PATH below. If a teammate hasn't
    downloaded the model yet, we raise a clear, friendly error instead
    of a confusing crash.
    """
    # Import here (not at the top of the file) so that the rest of the
    # app still works even if a teammate hasn't installed the audio
    # libraries yet - video support is the least commonly needed path.
    from moviepy import VideoFileClip
    import wave
    import json
    from vosk import Model, KaldiRecognizer

    # Folder where the downloaded Vosk model should live. Team should
    # download a small English model (e.g. "vosk-model-small-en-us")
    # from https://alphacephei.com/vosk/models and unzip it here.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vosk_model_path = os.path.join(project_root, "models", "vosk-model-small-en-us")

    if not os.path.isdir(vosk_model_path):
        raise InputExtractionError(
            "Speech-to-text model not found. Please download a Vosk "
            "English model from https://alphacephei.com/vosk/models, "
            f"unzip it, and place it at: {vosk_model_path}"
        )

    # Step 1: extract audio from the video into a temporary .wav file.
    temp_audio_path = os.path.join(tempfile.gettempdir(), "extracted_audio.wav")
    try:
        video = VideoFileClip(file_path)
        # Vosk expects mono, 16kHz audio - we set that here.
        video.audio.write_audiofile(
            temp_audio_path, fps=16000, nbytes=2, codec="pcm_s16le", logger=None
        )
        video.close()
    except Exception as error:
        raise InputExtractionError(f"Could not extract audio from video: {error}")

    # Step 2: run Vosk speech-to-text on the extracted audio.
    try:
        wave_file = wave.open(temp_audio_path, "rb")
        model = Model(vosk_model_path)
        recognizer = KaldiRecognizer(model, wave_file.getframerate())

        transcript_parts = []
        while True:
            data = wave_file.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                transcript_parts.append(result.get("text", ""))

        # Grab any leftover words after the loop ends.
        final_result = json.loads(recognizer.FinalResult())
        transcript_parts.append(final_result.get("text", ""))
        wave_file.close()

    except Exception as error:
        raise InputExtractionError(f"Speech-to-text failed: {error}")
    finally:
        # Clean up the temporary audio file either way.
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    full_transcript = " ".join(part for part in transcript_parts if part).strip()

    if not full_transcript:
        raise InputExtractionError(
            "No speech could be detected in this video's audio."
        )

    return full_transcript


# Lets a teammate quickly test this file directly, e.g.:
#   python core/input_handler.py sample.pdf
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python core/input_handler.py <file_path_or_text>")
    else:
        try:
            result_text = extract_text(sys.argv[1])
            print("--- Extracted Text ---")
            print(result_text)
        except InputExtractionError as e:
            print(f"[input_handler] Error: {e}")
