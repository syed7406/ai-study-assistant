from __future__ import annotations

from typing import Any, Dict, List

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from llm_client import build_answer
from llm_client import check_ollama_connection
from rag_pipeline import calculate_confidence, ingest_document, retrieve_relevant_chunks
from vector_store import delete_all_documents


app = FastAPI(title="AI Study Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)


class ClearRequest(BaseModel):
    user_id: str = Field(..., min_length=1)


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    idx = frontend_dist / "index.html" if frontend_dist.exists() else None
    if idx and idx.exists():
        return idx.read_text()
    return "<h1>AI Study Assistant</h1><p>Build frontend with <code>npm run build</code> in the frontend directory.</p>"


@app.get("/test", response_class=HTMLResponse)
def test_page() -> str:
    return """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>AI Study Assistant</title>
<style>body{font-family:sans-serif;background:#0b1020;color:#e8eefc;padding:2rem;max-width:800px;margin:0 auto}
.card{background:rgba(11,16,32,.72);border:1px solid rgba(255,255,255,.08);border-radius:20px;padding:1.25rem;margin-top:1rem}
button{border:0;border-radius:12px;padding:.85rem 1.1rem;background:linear-gradient(135deg,#6d7cff,#8c5cff);color:#fff;cursor:pointer;margin:5px}
input,select{flex:1;border-radius:12px;border:1px solid rgba(255,255,255,.12);background:rgba(255,255,255,.06);color:#fff;padding:.9rem 1rem;width:100%;box-sizing:border-box;margin:5px 0}
pre{background:rgba(0,0,0,.4);padding:1rem;border-radius:12px;overflow:auto}
#files li{cursor:pointer;padding:5px 10px;border-radius:8px;background:rgba(109,124,255,.15);color:#9bb1ff;margin:3px 0}
#files li:hover{background:rgba(109,124,255,.3)}.error{color:#ff6b6b}.success{color:#7cf0a8}</style>
</head><body>
<h1>AI Study Assistant</h1>
<div class="card">
  <h3>Upload a document</h3>
  <input type="file" id="fileInput">
  <button onclick="upload()">Upload</button>
  <div id="uploadResult"></div>
</div>
<div class="card">
  <h3>Your documents</h3>
  <ul id="files"></ul>
  <div id="content"></div>
</div>
<div class="card">
  <h3>Ask a question</h3>
  <input id="questionInput" placeholder="Ask a question..." onkeydown="if(event.key==='Enter')ask()">
  <button onclick="ask()">Ask</button>
  <div id="askResult"></div>
</div>
<script>
const U='demo-user';
function $(id){return document.getElementById(id)}
function e(t){return document.createElement(t)}
async function upload(){
  const f=$('fileInput').files[0];if(!f)return;
  const fd=new FormData();fd.append('file',f);fd.append('user_id',U);
  $('uploadResult').innerHTML='Uploading...';
  try{
    const r=await fetch('/upload',{method:'POST',body:fd});
    const d=await r.json();
    $('uploadResult').innerHTML=r.ok ? '<span class="success">'+d.chunks_stored+' chunks stored</span>' : '<span class="error">'+(d.detail||'Error')+'</span>';
    loadFiles();
  }catch(e){$('uploadResult').innerHTML='<span class="error">'+e.message+'</span>';}
}
async function loadFiles(){
  try{
    const r=await fetch('/documents?request_user_id='+U);const d=await r.json();
    const ul=$('files');ul.innerHTML='';
    if(!d.documents||!d.documents.length){ul.innerHTML='<li style="color:#888">No documents uploaded yet.</li>';return}
    d.documents.forEach(f=>{
      const li=e('li');li.textContent=f.filename+' ('+f.chunks+' ch)';
      li.onclick=()=>showContent(f.filename);ul.appendChild(li);
    });
  }catch(e){}
}
async function showContent(fn){
  $('content').innerHTML='<pre>Loading...</pre>';
  try{
    const r=await fetch('/documents/content?request_user_id='+U+'&filename='+encodeURIComponent(fn));
    const d=await r.json();
    $('content').innerHTML='<h4>'+fn+'</h4><pre>'+(d.content||'No content')+'</pre>';
  }catch(e){$('content').innerHTML='<pre class="error">Failed: '+e.message+'</pre>';}
}
async function ask(){
  const q=$('questionInput').value;if(!q)return;
  $('askResult').innerHTML='Thinking...';
  try{
    const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,user_id:U})});
    const d=await r.json();
    $('askResult').innerHTML='<p><strong>Answer:</strong> '+d.answer+'</p><p><strong>Confidence:</strong> '+d.confidence+'%</p>';
    if(d.sources&&d.sources.length){let h='<ul>';d.sources.forEach(s=>{h+='<li><strong>'+s.filename+'</strong>: '+s.excerpt.substring(0,100)+'...</li>'});h+='</ul>';$('askResult').innerHTML+=h}
  }catch(e){$('askResult').innerHTML='<span class="error">'+e.message+'</span>';}
}
loadFiles();
</script></body></html>"""


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/llm-status")
def llm_status() -> Dict[str, Any]:
    return check_ollama_connection()


MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), user_id: str = Form(...)) -> Dict[str, Any]:
    import logging
    logger = logging.getLogger("upload")
    logger.warning(f"UPLOAD START: file={file.filename} size={file.content_type} user={user_id}")

    file_bytes = await file.read()
    logger.warning(f"UPLOAD READ: {len(file_bytes)} bytes from {file.filename}")

    if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 20MB.")
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        chunks_stored = ingest_document(user_id, file_bytes, file.filename)
    except ValueError as exc:
        logger.warning(f"UPLOAD ERROR: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.warning(f"UPLOAD OK: {file.filename} -> {chunks_stored} chunks")

    if chunks_stored == 0:
        raise HTTPException(
            status_code=400,
            detail=f"No extractable text found in '{file.filename}'. "
            "If this is a scanned PDF or image, please use a text-based file.",
        )

    return {
        "status": "success",
        "chunks_stored": chunks_stored,
        "filename": file.filename,
        "user_id": user_id,
    }


@app.get("/documents")
def list_documents(request_user_id: str = "") -> Dict[str, Any]:
    from vector_store import get_collection
    try:
        col = get_collection(request_user_id if request_user_id else None)
        count = col.count()
        if count == 0:
            return {"documents": [], "count": 0, "message": "No documents uploaded yet."}
        results = col.get(include=["metadatas"])
        files = {}
        for meta in results.get("metadatas", []):
            fn = meta.get("filename", "unknown")
            files[fn] = files.get(fn, 0) + 1
        return {"documents": [{"filename": k, "chunks": v} for k, v in files.items()], "count": count}
    except Exception:
        return {"documents": [], "count": 0, "message": "No documents uploaded yet."}


@app.get("/documents/content")
def get_document_content(request_user_id: str = "", filename: str = "") -> Dict[str, Any]:
    from vector_store import get_collection
    if not request_user_id or not filename:
        return {"content": "", "chunks": []}
    try:
        col = get_collection(request_user_id)
        results = col.get(include=["documents", "metadatas"])
        chunks = []
        for doc, meta in zip(results.get("documents", []), results.get("metadatas", [])):
            if meta.get("filename") == filename:
                chunks.append({
                    "section": meta.get("section", 1),
                    "text": doc,
                })
        full_text = "\n\n".join(c["text"] for c in chunks)
        return {"content": full_text, "chunks": chunks, "filename": filename}
    except Exception:
        return {"content": "", "chunks": []}


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
                "page": metadata.get("page", metadata.get("section", "unknown")),
                "excerpt": text[:240],
            }
        )

    if not documents:
        return {
            "answer": "No relevant information found in your uploaded documents.",
            "confidence": 0.0,
            "sources": [],
        }

    return {"answer": answer, "confidence": confidence, "sources": sources}


@app.delete("/documents")
def clear_documents(request: ClearRequest) -> Dict[str, str]:
    delete_all_documents(request.user_id)
    return {"status": "cleared", "user_id": request.user_id}
