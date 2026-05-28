# Supported file types

The document loader (`agent/ingestion/loader.py`) supports these extensions:

| Extension | Parser | Notes |
|-----------|--------|-------|
| `.pdf` | `pypdf` | Each page becomes a separate unit with its page number in metadata. Text-based PDFs only; scanned images are not OCR-processed. |
| `.txt` | built-in | Read as UTF-8; undecodable bytes are ignored. |
| `.docx` | `python-docx` | Non-empty paragraphs are concatenated. |

Unsupported extensions are skipped during directory loading. Retrieval quality
depends on parsing quality, so clean, text-based documents work best.
