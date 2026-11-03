"""Management endpoints protected by the existing LLM internal boundary."""
from __future__ import annotations

from typing import Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException

from services.llm.auth import authorize_llm_internal_request
from services.llm.summary.repository import SummaryRepository
from services.llm.summary.schema import SummaryPreferences


def summary_router(repository: SummaryRepository) -> APIRouter:
    async def available() -> None:
        if not repository.settings.enabled:
            raise HTTPException(503, "Summary feature is disabled")
        try:
            await repository.check_schema()
        except Exception:
            raise HTTPException(503, "Summary storage unavailable or migration missing") from None

    router = APIRouter(prefix="/api/v1/llm/memory/{user_id}/summary",
                       dependencies=[Depends(authorize_llm_internal_request), Depends(available)])

    @router.get("")
    async def read(user_id: str) -> dict[str, Any]:
        return await repository.read(user_id)

    @router.get("/settings")
    async def read_settings(user_id: str) -> dict[str, Any]:
        current = await repository.read(user_id)
        config = repository.settings
        return {"enabled": current["enabled"], "apply_to_prompt": current["apply_to_prompt"],
                "global_enabled": config.enabled, "global_apply_to_prompt": config.apply_to_prompt,
                "effective_generation": config.enabled and current["enabled"],
                "effective_apply_to_prompt": config.enabled and config.apply_to_prompt
                    and current["enabled"] and current["apply_to_prompt"],
                "summary_status": current["status"]}

    @router.post("/refresh", status_code=202)
    async def refresh(user_id: str) -> dict[str, Any]:
        await repository.state(user_id)
        job = await repository.enqueue(user_id)
        if not job:
            raise HTTPException(409, "No eligible turns or summaries disabled for this user")
        return {"job_id": str(job["job_id"]), "status": job["status"]}

    @router.get("/jobs/{job_id}")
    async def job_status(user_id: str, job_id: UUID) -> dict[str, Any]:
        job = await repository.job(user_id, job_id)
        if not job:
            raise HTTPException(404, "Summary job not found")
        return job

    @router.delete("")
    async def delete(user_id: str) -> dict[str, Any]:
        return await repository.reset(user_id)

    @router.patch("/settings")
    async def settings(user_id: str, body: SummaryPreferences) -> dict[str, Any]:
        changes = body.model_dump(exclude_none=True)
        if not changes:
            return await repository.read(user_id)
        return await repository.reset(user_id, changes)

    return router
