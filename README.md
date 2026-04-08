# AI Legal Document Analyzer

Streamlit app for analyzing legal PDFs with Google Gemini.

It sends the uploaded PDF directly to Gemini for:
- document summary
- clause detection
- risk detection
- people and organization extraction
- overall risk scoring

Local `pdfplumber` extraction is only used for optional stats like page count and word count when text is available.

## Files

```text
app.py
extractor.py
analyzer.py
utils.py
requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

## Run

```bash
streamlit run app.py
```

## Notes

- Scanned or image-based PDFs are supported through Gemini PDF analysis.
- Local word count may be low or zero when the PDF has no embedded text, but Gemini can still analyze the document itself.
- The analyzer uses `gemini-2.5-flash` with structured JSON output.
