"""FastAPI entry point for knowledge ingestion and GraphRAG retrieval."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from services.knowledge.auth import authorize_knowledge_admin_request, authorize_knowledge_internal_request
from services.knowledge.config import load_settings
from services.knowledge.schemas import (
    KnowledgeContext,
    KnowledgeDocumentRequest,
    KnowledgeDocumentResponse,
    KnowledgeRetrieveRequest,
    KnowledgeStructuredDocumentRequest,
)
from services.knowledge.service import KnowledgeService


settings = load_settings()
knowledge_service = KnowledgeService(settings)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    await knowledge_service.initialize()
    try:
        yield
    finally:
        await knowledge_service.close()


app = FastAPI(title="Nawei Knowledge GraphRAG Service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> JSONResponse:
    status = await knowledge_service.health()
    body = {"status": "ok" if status["ready"] else "degraded", **status}
    return JSONResponse(status_code=200 if status["ready"] else 503, content=body)


@app.post(
    "/api/v1/knowledge/documents/text",
    response_model=KnowledgeDocumentResponse,
    dependencies=[Depends(authorize_knowledge_admin_request)],
)
async def ingest_text(request: KnowledgeDocumentRequest) -> KnowledgeDocumentResponse:
    try:
        return await knowledge_service.ingest(request)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post(
    "/api/v1/knowledge/documents/structured",
    response_model=KnowledgeDocumentResponse,
    dependencies=[Depends(authorize_knowledge_admin_request)],
)
async def ingest_structured(request: KnowledgeStructuredDocumentRequest) -> KnowledgeDocumentResponse:
    try:
        return await knowledge_service.ingest_structured(request)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post(
    "/api/v1/knowledge/documents/upload",
    response_model=KnowledgeDocumentResponse,
    dependencies=[Depends(authorize_knowledge_admin_request)],
)
async def upload_document(
    file: UploadFile = File(...),
    knowledge_base_id: str = Form("default"),
    title: str | None = Form(None),
) -> KnowledgeDocumentResponse:
    filename = file.filename or "knowledge.txt"
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in {"txt", "md", "markdown"}:
        raise HTTPException(status_code=415, detail="The first version supports TXT and Markdown files only.")
    raw = await file.read(2_000_001)
    if len(raw) > 2_000_000:
        raise HTTPException(status_code=413, detail="Knowledge document exceeds the 2 MB limit.")
    content = _decode_text(raw)
    request = KnowledgeDocumentRequest(
        knowledge_base_id=knowledge_base_id,
        title=title or filename,
        source_name=filename,
        content=content,
        metadata={"content_type": file.content_type or "text/plain"},
    )
    try:
        return await knowledge_service.ingest(request)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post(
    "/api/v1/knowledge/retrieve",
    response_model=KnowledgeContext,
    dependencies=[Depends(authorize_knowledge_internal_request)],
)
async def retrieve(request: KnowledgeRetrieveRequest) -> KnowledgeContext:
    try:
        return await knowledge_service.retrieve(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get(
    "/api/v1/knowledge/documents/{document_id}",
    dependencies=[Depends(authorize_knowledge_admin_request)],
)
async def document_status(document_id: str) -> dict[str, Any]:
    result = await knowledge_service.get_document(document_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Knowledge document was not found.")
    return result


@app.delete(
    "/api/v1/knowledge/documents/{document_id}",
    dependencies=[Depends(authorize_knowledge_admin_request)],
)
async def delete_document(document_id: str) -> dict[str, bool]:
    return {"deleted": await knowledge_service.delete_document(document_id)}


def _decode_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise HTTPException(status_code=400, detail="Unable to decode the uploaded text file.")
