# AI Legal Document Analyzer

Streamlit app for analyzing legal PDFs with a local Ollama model.

It extracts text from the uploaded PDF and analyzes it in a small-prompt pipeline:
- split document text into overlapping chunks
- run chunk-level extraction for summary, entities, clauses, and risks
- merge and deduplicate the chunk evidence
- run a final synthesis pass for the document-level result

The final output still includes:
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
OLLAMA_MODEL=gemma4
OLLAMA_TIMEOUT_SECONDS=0
MAX_ANALYSIS_CHARS=120000
OLLAMA_CHUNK_SIZE_CHARS=6000
OLLAMA_CHUNK_OVERLAP_CHARS=500
OLLAMA_MAX_CHUNKS=24
```

## Run

```bash
streamlit run app.py
```

## Notes

- The default Ollama endpoint is `http://localhost:11434`.
- The default model is `gemma4`.
- `OLLAMA_TIMEOUT_SECONDS=0` disables the client-side timeout; set a positive number of seconds if you want a limit.
- The analyzer now uses a chunked pipeline so smaller local models do not need to handle the full document and all tasks in one prompt.
- Reduce `OLLAMA_CHUNK_SIZE_CHARS` if your local model still struggles with context or latency.
- Scanned or image-only PDFs will need OCR before analysis in the current implementation.
