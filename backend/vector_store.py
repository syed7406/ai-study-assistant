from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings


@lru_cache(maxsize=1)
def get_collection_name() -> str:
    return "study_assistant_documents"


def get_user_collection_name(user_id: str) -> str:
    safe_user_id = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)
    return f"{get_collection_name()}_{safe_user_id}"


@lru_cache(maxsize=1)
def get_client() -> chromadb.PersistentClient:
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(user_id: str | None = None):
    client = get_client()
    collection_name = get_collection_name() if user_id is None else get_user_collection_name(user_id)
    return client.get_or_create_collection(name=collection_name)


def add_documents(
    user_id: str,
    ids: List[str],
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    embeddings: List[List[float]],
):
    collection = get_collection(user_id)
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)


def query_documents(user_id: str, query_embedding: List[float], top_k: int):
    collection = get_collection(user_id)
    return collection.query(query_embeddings=[query_embedding], n_results=top_k, include=["documents", "metadatas", "distances"])


def delete_all_documents(user_id: str):
    client = get_client()
    try:
        client.delete_collection(name=get_user_collection_name(user_id))
    except Exception:
        pass
