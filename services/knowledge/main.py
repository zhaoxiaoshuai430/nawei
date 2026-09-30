"""FastAPI entry point for knowledge ingestion and GraphRAG retrieval."""

# Knowledge 的接口层只做鉴权、输入限制及 HTTP 错误转换，索引/检索流程在 service.py。
# 入库、删除和状态查询使用管理员鉴权；LLM 调用 retrieve 使用内部服务鉴权。
# 返回知识片段、关系和来源，不在此生成最终对话，也不存储用户聊天记忆。

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
    # 在接请求前连接存储并预热本地向量模型；停机时统一释放连接和模型提供者资源。
    await knowledge_service.initialize()
    try:
        yield
    finally:
        await knowledge_service.close()


app = FastAPI(title="Nawei Knowledge GraphRAG Service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> JSONResponse:
    # 区分“进程活着”和“依赖已就绪”：未就绪返回 503，供启动脚本和健康探测判断。
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
    # 结构化导入接收已整理的片段、引用定位和显式关系，保留人工知识包的结构。
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
    # 当前只接收文本/Markdown，不是通用 PDF/Excel 解析接口。
    # 多读取一个字节用于识别超限文件，避免无界读取后才检查上传大小。
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
    # request 携带知识库范围、问题以及可选概念锚点；返回证据上下文供 LLM 组织回答。
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
