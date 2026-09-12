# VaultMind v2 — RAG + Projects + SQLite

Full-stack RAG chat app with persistent Projects mode, SQLite storage, and a glassmorphism UI.

## Stack

| Layer | Tool |
|---|---|
| Backend | Flask (Python) |
| Database | SQLite (built-in, zero setup) |
| LLM | Groq `llama3-8b-8192` |
| Embeddings | `all-MiniLM-L6-v2` (local, free) |
| Vector Store | ChromaDB (per-project / per-conversation) |
| Retrieval | MMR (Maximal Marginal Relevance) |
| Frontend | Vanilla JS + Glassmorphism UI |

## Setup

```bash
# 1. Clone / unzip into a folder
cd vaultmind/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Groq API key (free at console.groq.com)
echo "GROQ_API_KEY=gsk_your_key_here" > .env

# 4. Run
python app.py
```

Then open **http://localhost:5000** in your browser.

## Features

### Projects mode (like Claude)
- Create named projects with a description and custom instructions
- Upload documents to a project's **Knowledge Base** — they're shared across all conversations in that project
- Each project's conversations automatically use its RAG context
- All projects, conversations, and messages persist in `vaultmind.db` (SQLite)

### Standalone conversations
- Chat without a project
- Upload documents per-conversation
- History persists in SQLite across restarts

### RAG pipeline
- Documents are chunked and embedded into ChromaDB (one collection per project / per conversation)
- Queries use MMR retrieval → Groq LLM → cited answer
- If no documents are uploaded, falls back to plain LLM chat

## File structure

```
vaultmind/
  app.py              ← Flask backend (SQLite + RAG + API)
  requirements.txt
  .env                ← your GROQ_API_KEY
  static/
    index.html        ← Full frontend (Projects UI + chat)
  uploads/            ← Uploaded files (auto-created)
    project_1/
    conv_3/
  chroma_db/          ← Vector stores (auto-created)
    project_1/
    conv_3/
  vaultmind.db        ← SQLite database (auto-created on first run)
```

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/projects` | List all projects |
| POST | `/api/projects` | Create project |
| PUT | `/api/projects/:id` | Edit project |
| DELETE | `/api/projects/:id` | Delete project + its data |
| GET | `/api/projects/:id/files` | List project files |
| POST | `/api/projects/:id/files` | Upload file to project |
| DELETE | `/api/projects/:id/files/:fid` | Remove file |
| GET | `/api/projects/:id/index-status` | Check indexing progress |
| GET | `/api/conversations` | List conversations (all or by project) |
| POST | `/api/conversations` | Create conversation |
| DELETE | `/api/conversations/:id` | Delete conversation |
| GET | `/api/conversations/:id/messages` | Load message history |
| POST | `/api/conversations/:id/messages` | Send message → RAG response |
| GET | `/api/conversations/:id/files` | List conversation files |
| POST | `/api/conversations/:id/files` | Upload file to conversation |

## Tips

- **First run** downloads the embedding model (~90 MB) — subsequent starts are instant
- **Longer docs**: increase `CHUNK_SIZE` in `app.py` to 1500–2000
- **Different model**: swap `GROQ_MODEL` to `mixtral-8x7b-32768` for 32k context
- **Reset a project's index**: delete `chroma_db/project_<id>/` and re-upload files
- **Production**: run with `gunicorn -w 1 -b 0.0.0.0:5000 app:app` (single worker for ChromaDB safety)

# vaultmind-RAG
