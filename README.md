# AI Legal Document Analyzer

Streamlit app for analyzing legal PDFs with a local Ollama model.

It extracts text from the uploaded PDF and sends that text to Ollama for:
- document summary
- clause detection
- risk detection
- people and organization extraction
- overall risk scoring

If no text can be extracted, the app currently stops with a clear error because this Ollama path does not handle raw scanned PDFs directly.

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

Install and run Ollama separately, then make sure your model is pulled:

```bash
ollama pull llama3.1:8b
ollama serve
```

Optional `.env` settings:

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3.5:9b
OLLAMA_TIMEOUT_SECONDS=0
```

## Run

```bash
streamlit run app.py
```

## Notes

- The default Ollama endpoint is `http://localhost:11434`.
- The default model is `qwen3.5:9b`.
- `OLLAMA_TIMEOUT_SECONDS=0` disables the client-side timeout; set a positive number of seconds if you want a limit.
- Scanned or image-only PDFs will need OCR before analysis in the current implementation.
