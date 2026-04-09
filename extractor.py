"""
extractor.py
------------
Production-grade PDF text extraction for legal documents.

Strategy: Surya OCR on every page, always — no pdfplumber shortcuts.
Legal documents demand zero tolerance for missed text, so every page
is rasterised and passed through the full preprocessing + OCR pipeline
regardless of whether a text layer exists.

Preprocessing pipeline (applied per page before OCR):
  1. Convert to grayscale
  2. Upscale if resolution is too low (< 300 DPI equivalent)
  3. CLAHE contrast enhancement (handles faded ink, uneven lighting)
  4. Denoising (removes scanner artefacts, JPEG noise)
  5. Otsu binarization (clean black-on-white)
  6. Deskew (fixes photographed / rotated pages up to ±15°)
  7. Border removal (strips black scan borders that confuse OCR)
  8. Morphological cleanup (closes small gaps in characters)

Install
-------
  pip install pdf2image surya-ocr opencv-python-headless pillow numpy

  System deps for pdf2image:
    Linux : apt install poppler-utils
    macOS : brew install poppler
    Win   : https://github.com/oschwartz10612/poppler-windows

  Surya downloads model weights on first run (~1-2 GB). Cached after that.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from pdf2image import convert_from_bytes
from PIL import Image

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

# DPI for rasterisation. 300 is the OCR standard.
# Use 400 for documents with very small fonts (< 8pt).
RASTERISE_DPI = 300

# If a page image is narrower than this after rasterisation,
# upscale it before OCR — some scanned docs come in at low resolution.
MIN_WIDTH_PX = 1200

# Surya language hints. None = auto-detect (works well for English + Indic).
# Override with e.g. ["en", "hi", "mr"] if auto-detect is inconsistent.
SURYA_LANGS = None


# ── Public API ────────────────────────────────────────────────────────────────

def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract all text from a PDF using Surya OCR.

    Every page is rasterised and OCR'd — no text-layer shortcuts.
    This guarantees consistent accuracy across digital PDFs, scanned
    documents, and photographed contracts.

    Args:
        uploaded_file: A file-like object (Streamlit file_uploader, open(),
                       BytesIO, or raw bytes).

    Returns:
        Full document text as a single string, pages separated by double
        newlines. Raises RuntimeError if extraction fails entirely.
    """
    raw_bytes = _read_bytes(uploaded_file)

    logger.info("Rasterising PDF at %d DPI...", RASTERISE_DPI)
    images = _rasterise_pdf(raw_bytes)
    if not images:
        raise RuntimeError(
            "Could not rasterise PDF. Ensure poppler-utils is installed "
            "and the file is a valid PDF."
        )

    logger.info("Preprocessing %d page(s)...", len(images))
    preprocessed = [_preprocess(img, page_num=i + 1) for i, img in enumerate(images)]

    logger.info("Running Surya OCR on %d page(s)...", len(preprocessed))
    text = _ocr_with_surya(preprocessed)

    if not text.strip():
        raise RuntimeError(
            "Surya OCR returned no text. The PDF may be blank, "
            "fully redacted, or corrupted."
        )

    logger.info(
        "Extraction complete: %d words across %d page(s).",
        len(text.split()), len(images),
    )
    return text


def get_page_count(uploaded_file) -> int:
    """Return the number of pages in the PDF."""
    raw_bytes = _read_bytes(uploaded_file)
    images = _rasterise_pdf(raw_bytes)
    return len(images)


# ── Rasterisation ─────────────────────────────────────────────────────────────

def _rasterise_pdf(raw_bytes: bytes) -> list[Image.Image]:
    """Rasterise every PDF page to a high-resolution PIL Image."""
    try:
        return convert_from_bytes(
            raw_bytes,
            dpi=RASTERISE_DPI,
            fmt="png",           # lossless — no JPEG artefacts
            thread_count=4,      # parallel page rendering
            use_cropbox=True,    # respect cropbox for correct page size
            strict=False,        # tolerate minor PDF malformations
        )
    except Exception as exc:
        logger.error("PDF rasterisation failed: %s", exc)
        return []


# ── Preprocessing pipeline ────────────────────────────────────────────────────

def _preprocess(pil_image: Image.Image, page_num: int = 1) -> Image.Image:
    """
    Apply the full preprocessing pipeline to a single page image.
    Returns a cleaned, binarised PIL Image ready for Surya OCR.
    """
    try:
        import cv2

        img = np.array(pil_image.convert("RGB"))

        # Step 1: Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Step 2: Upscale if image is too small
        h, w = gray.shape
        if w < MIN_WIDTH_PX:
            scale = MIN_WIDTH_PX / w
            gray = cv2.resize(
                gray,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )
            logger.debug(
                "Page %d: upscaled %.1fx to %dpx wide.", page_num, scale, int(w * scale)
            )

        # Step 3: CLAHE — adaptive contrast enhancement
        # Handles faded ink, yellowed paper, uneven scanner lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Step 4: Denoising
        # h=10 removes noise without blurring character edges
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

        # Step 5: Otsu binarization
        # Automatically finds the best threshold for black-on-white text
        _, binary = cv2.threshold(
            denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Step 6: Deskew
        binary = _deskew(binary, page_num)

        # Step 7: Remove black scan borders
        binary = _remove_borders(binary)

        # Step 8: Morphological cleanup
        # Closes tiny gaps inside characters (e.g. broken ink on old documents)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return Image.fromarray(binary)

    except ImportError:
        logger.warning(
            "opencv not installed — preprocessing skipped for page %d.", page_num
        )
        return pil_image
    except Exception as exc:
        logger.warning(
            "Preprocessing failed on page %d (%s) — using raw image.", page_num, exc
        )
        return pil_image


def _deskew(binary: np.ndarray, page_num: int = 1) -> np.ndarray:
    """
    Detect and correct skew using minAreaRect on foreground pixel coordinates.

    Corrects angles between 0.3° and 15°. Outside this range:
    - < 0.3° : negligible, skip
    - > 15°  : probably intentional rotation (landscape page), skip

    For pages rotated 90° (portrait scanned as landscape), Surya handles
    this internally via its layout detection.
    """
    try:
        import cv2

        # Foreground = dark pixels (text)
        coords = np.column_stack(np.where(binary < 128))
        if len(coords) < 200:
            return binary  # not enough content to estimate angle reliably

        angle = cv2.minAreaRect(coords)[-1]

        # minAreaRect returns [-90, 0); remap to [-45, 45)
        if angle < -45:
            angle = 90 + angle

        if abs(angle) < 0.3:
            return binary  # negligible skew
        if abs(angle) > 15:
            logger.debug(
                "Page %d: skew angle %.2f° exceeds threshold — skipping deskew.",
                page_num,
                angle,
            )
            return binary

        h, w = binary.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            binary,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        logger.debug("Page %d: deskewed by %.2f°.", page_num, angle)
        return rotated

    except Exception as exc:
        logger.warning("Deskew failed on page %d: %s", page_num, exc)
        return binary


def _remove_borders(binary: np.ndarray, border_fraction: float = 0.02) -> np.ndarray:
    """
    Remove dark scanner borders from the edges of the image.

    Many scanned documents have a black or very dark border artefact around
    the edge. This crops a small fraction of the image on each side to remove
    it, preventing Surya from misreading border artefacts as text.

    border_fraction=0.02 crops 2% from each edge — safe for most scanners.
    """
    h, w = binary.shape
    margin_h = max(1, int(h * border_fraction))
    margin_w = max(1, int(w * border_fraction))
    return binary[margin_h : h - margin_h, margin_w : w - margin_w]


# ── Surya OCR ─────────────────────────────────────────────────────────────────

_surya_predictors: dict | None = None


def _load_surya_predictors() -> dict:
    global _surya_predictors
    if _surya_predictors is not None:
        return _surya_predictors

    logger.info("Loading Surya predictors (first run may download weights)...")

    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor

    foundation = FoundationPredictor()

    _surya_predictors = {
        "recognition": RecognitionPredictor(foundation),
        "detection":   DetectionPredictor(),
    }

    logger.info("Surya predictors loaded successfully.")
    return _surya_predictors


def _ocr_with_surya(images: list[Image.Image]) -> str:
    """
    Run Surya OCR on a list of preprocessed PIL images.
    """
    predictors = _load_surya_predictors()

    rec = predictors["recognition"]
    det = predictors["detection"]

    # Removed the langs list entirely because Surya 0.17.0+ handles it automatically
    results = rec(images, det_predictor=det)

    page_texts: list[str] = []
    for page_num, page_result in enumerate(results, start=1):
        lines = []
        for line in page_result.text_lines:
            text = line.text.strip()
            if not text:
                continue
            if hasattr(line, "confidence") and line.confidence < 0.3:
                logger.debug(
                    "Page %d: dropped low-confidence line (%.2f): %r",
                    page_num, line.confidence, text,
                )
                continue
            lines.append(text)

        page_text = "\n".join(lines)
        page_texts.append(page_text)
        logger.debug(
            "Page %d: %d line(s), %d word(s).",
            page_num, len(lines), len(page_text.split()),
        )

    return "\n\n".join(page_texts)

# ── Internal helpers ──────────────────────────────────────────────────────────

def _read_bytes(source) -> bytes:
    """Normalise various input types to raw bytes."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if hasattr(source, "read"):
        data = source.read()
        if hasattr(source, "seek"):
            source.seek(0)  # reset so caller can re-read if needed
        return data
    raise TypeError(f"Cannot read bytes from {type(source)}")



