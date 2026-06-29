# AI Study Assistant with RAG + Voice

Full-stack AI study app that lets users upload PDFs, ask questions grounded strictly in their documents, and use voice input in the browser.

## Tech Stack

- Frontend: React + Vite
- Backend: FastAPI + LangChain-style RAG pipeline
- Vector DB: ChromaDB
- Embeddings: sentence-transformers
- PDF extraction: PyMuPDF
- LLM: Ollama by default, with optional Gemini fallback
- Auth: Firebase-ready placeholders
- Deployment: Docker + Render-ready backend

## Project Structure

- `frontend/` React app
- `backend/` FastAPI app
- `docker-compose.yml` Local orchestration

## Run Locally

### Backend

1. Create a virtual environment.
2. Install dependencies from `backend/requirements.txt`.
3. Copy `backend/.env.example` to `backend/.env`.
4. Start Ollama locally if you want local LLM generation.
5. Start the API with Uvicorn.

### Local LLM

- Default provider: Ollama
- Suggested model: `llama3.2`
- If Ollama is not running, the backend falls back to a grounded extractive answer.

### Frontend

1. Install dependencies from `frontend/package.json`.
2. Copy `frontend/.env.example` to `frontend/.env`.
3. Start the Vite dev server.

Frontend environment variables use the Vite `VITE_` prefix.

## API Endpoints

- `POST /upload`
- `POST /ask`
- `DELETE /documents`
- `GET /health`

## Notes

- The backend is scaffolded to answer only from retrieved document chunks.
- The backend uses a free-first design so you can demo without paid API keys.
- Firebase and hosted deployment values are placeholders and should be replaced.
