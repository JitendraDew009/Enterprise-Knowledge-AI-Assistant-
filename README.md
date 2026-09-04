# Enterprise Knowledge AI Assistant

A FastAPI service for asking grounded questions about enterprise documents. The application uses LangChain for document splitting, embeddings, vector retrieval, and LLM response generation.

## Features

- Upload `.txt`, `.md`, and `.pdf` documents.
- Split documents into retrievable chunks with source metadata.
- Store embeddings in a local Chroma vector database.
- Retrieve relevant context and generate source-aware answers with an OpenAI-compatible chat model.
- Run without API credentials in development using deterministic local embeddings and an extractive fallback.

## Quick start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/` for the browser assistant or
`http://127.0.0.1:8000/docs` for the interactive API documentation.

Set `OPENAI_API_KEY` in `.env` to enable LLM-generated answers. Without it, the API still retrieves context and returns a deterministic answer assembled from the most relevant passages.

## API

### Upload a document

```powershell
curl.exe -X POST http://127.0.0.1:8000/documents -F "file=@handbook.md"
```

### Ask a question

```powershell
curl.exe -X POST http://127.0.0.1:8000/query `
  -H "Content-Type: application/json" `
  -d '{"question":"What is our remote work policy?"}'
```

### List and remove documents

```powershell
curl.exe http://127.0.0.1:8000/documents
curl.exe -X DELETE http://127.0.0.1:8000/documents/handbook.md
```

Responses include the answer, retrieved source excerpts, and retrieval scores for evaluation.

## Development

```powershell
pytest
ruff check .
```

The local Chroma data directory is ignored by Git. For production, configure a managed vector database, authentication, upload limits, and persistent object storage before deploying.
