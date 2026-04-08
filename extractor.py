"""
extractor.py
------------
Handles PDF text extraction using pdfplumber.
"""

import pdfplumber


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract all text from an uploaded PDF file.

    Args:
        uploaded_file: A file-like object (from Streamlit's file_uploader).

    Returns:
        A single string containing all extracted text.
    """
    full_text = []

    # Open the PDF using pdfplumber
    with pdfplumber.open(uploaded_file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Extract text from each page
            page_text = page.extract_text()

            if page_text:
                full_text.append(page_text)

    # Join all page texts with newlines
    return "\n".join(full_text)


def get_page_count(uploaded_file) -> int:
    """
    Return the number of pages in the PDF.

    Args:
        uploaded_file: A file-like object.

    Returns:
        Integer page count.
    """
    with pdfplumber.open(uploaded_file) as pdf:
        return len(pdf.pages)
