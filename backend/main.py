from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm_client import build_answer
from llm_client import check_ollama_connection
from rag_pipeline import calculate_confidence, ingest_document, retrieve_relevant_chunks
from vector_store import delete_all_documents


app = FastAPI(title="AI Study Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)


class ClearRequest(BaseModel):
    user_id: str = Field(..., min_length=1)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/llm-status")
def llm_status() -> Dict[str, Any]:
    return check_ollama_connection()


MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), user_id: str = Form(...)) -> Dict[str, Any]:
    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 20MB.")
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        chunks_stored = ingest_document(user_id, file_bytes, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "success",
        "chunks_stored": chunks_stored,
        "filename": file.filename,
        "user_id": user_id,
    }


@app.post("/ask")
def ask(request: AskRequest) -> Dict[str, Any]:
    results = retrieve_relevant_chunks(request.user_id, request.question)
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    answer = build_answer(request.question, documents)
    confidence = calculate_confidence(distances)

    sources: List[Dict[str, Any]] = []
    for text, metadata in zip(documents, metadatas):
        sources.append(
            {
                "filename": metadata.get("filename", "unknown"),
                "page": metadata.get("page", "unknown"),
                "excerpt": text[:240],
            }
        )

    return {"answer": answer, "confidence": confidence, "sources": sources}


@app.delete("/documents")
def clear_documents(request: ClearRequest) -> Dict[str, str]:
    delete_all_documents(request.user_id)
    return {"status": "cleared", "user_id": request.user_id}
