"""Bocha web search tool for current web lookup."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import httpx


BOCHA_WEB_SEARCH_URL = "https://api.bochaai.com/v1/web-search"
DEFAULT_TIMEOUT_SECONDS = 8.0
DEFAULT_COUNT = 5
DEFAULT_CACHE_MAX_ENTRIES = 256
SEARCH_MARKERS = (
    "联网搜索",
    "全网搜索",
    "网上搜索",
    "网络搜索",
    "搜索",
    "搜一下",
    "搜搜",
    "网上搜",
    "查新闻",
    "新闻",
    "最新",
    "最近",
    "实时",
    "今年",
    "世界杯",
    "赛事",
    "比赛",
    "赛程",
    "冠军",
    "比分",
    "全网",
    "world cup",
    "web search",
    "search web",
    "bocha",
    "博查",
)
NON_SEARCH_MARKERS = (
    "天气",
    "气温",
    "温度",
    "下雨",
    "降雨",
    "带伞",
    "穿什么",
    "穿衣",
    "衣服",
    "单词",
    "英文",
    "英语",
    "怎么读",
    "计算",
    "算一下",
    "等于多少",
    "几点",
    "时间",
    "日期",
)
SEARCH_NOISE_PATTERN = re.compile(
    r"(?:请|麻烦|帮我|给我|你|一下|下|关于|有关|"
    r"联网搜索|全网搜索|网上搜索|网络搜索|联网|上网|网上|网络|搜索|搜一下|搜搜|搜|查新闻|查一下|查询|查找|检索|"
    r"全网|web search|search web|bocha|博查)",
    flags=re.I,
)


@dataclass(frozen=True)
class WebSearchItem:
    title: str
    url: str
    snippet: str = ""
    summary: str = ""
    site_name: str = ""
    date: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "summary": self.summary,
            "site_name": self.site_name,
            "date": self.date,
        }


@dataclass(frozen=True)
class WebSearchResult:
    query: str
    results: list[WebSearchItem] = field(default_factory=list)
    source: str = "bocha"
    resource_package_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "results": [item.to_dict() for item in self.results],
            "source": self.source,
            "resource_package_id": self.resource_package_id,
        }


class BochaWebSearchTool:
    def __init__(
        self,
        *,
        timeout_seconds: float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds or _read_float_env("BOCHA_SEARCH_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
        self.api_key = os.getenv("BOCHA_API_KEY", os.getenv("BOCHA_SEARCH_API_KEY", "")).strip()
        self.resource_package_id = os.getenv("BOCHA_RESOURCE_PACKAGE_ID", "").strip()
        self.endpoint = os.getenv("BOCHA_WEB_SEARCH_URL", BOCHA_WEB_SEARCH_URL).strip() or BOCHA_WEB_SEARCH_URL
        self.default_count = _read_int_env("BOCHA_SEARCH_COUNT", DEFAULT_COUNT)
        self.summary_enabled = _read_bool_env("BOCHA_SEARCH_SUMMARY_ENABLED", True)
        self._cache: dict[tuple[str, int, str], tuple[float, WebSearchResult]] = {}
        self.cache_ttl_seconds = _read_float_env("BOCHA_SEARCH_CACHE_TTL_SECONDS", 300.0)
        self.cache_max_entries = max(1, _read_int_env("BOCHA_SEARCH_CACHE_MAX_ENTRIES", DEFAULT_CACHE_MAX_ENTRIES))
        self._client = client
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _http_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_seconds, trust_env=False)
        return self._client

    def extract_query(self, text: str) -> str | None:
        normalized = _normalize_text(text)
        if not normalized:
            return None
        if any(marker.lower() in normalized.lower() for marker in NON_SEARCH_MARKERS):
            return None
        if not any(marker.lower() in normalized.lower() for marker in SEARCH_MARKERS):
            return None
        query = _clean_search_query(normalized)
        return query if query and len(query) >= 2 else None

    async def lookup_from_text(self, text: str) -> WebSearchResult | None:
        query = self.extract_query(text)
        if not query:
            return None
        return await self.lookup(query=query)

    async def lookup(self, *, query: str, count: int | None = None, freshness: str | None = None) -> WebSearchResult:
        if not self.api_key:
            raise ValueError("BOCHA_API_KEY is required for Bocha web search")
        query = query.strip()
        if not query:
            raise ValueError("Search query is required")
        count = max(1, min(int(count or self.default_count), 10))
        cache_key = (query.lower(), count, (freshness or "").strip().lower())
        now = time.monotonic()
        self._prune_cache(now)
        cached = self._cache.get(cache_key)
        if cached and self.cache_ttl_seconds > 0 and now - cached[0] <= self.cache_ttl_seconds:
            return cached[1]

        queries = _expanded_world_cup_result_queries(query) + [query]
        results: list[WebSearchItem] = []
        last_error: Exception | None = None
        for current_query in _unique_queries(queries):
            try:
                partial = await self._lookup_once(
                    client=self._http_client(),
                    query=current_query,
                    count=count,
                    freshness=freshness,
                )
                results.extend(partial.results)
            except Exception as exc:
                last_error = exc
                continue
        merged = _dedupe_search_items(results)
        if not merged:
            if last_error is not None:
                raise last_error
            raise ValueError("Bocha web search returned no results")
        result = WebSearchResult(query=query, results=merged[:count], resource_package_id=self.resource_package_id)
        if self.cache_ttl_seconds > 0:
            self._cache[cache_key] = (now, result)
            self._prune_cache(now)
        return result

    def _prune_cache(self, now: float) -> None:
        if self.cache_ttl_seconds > 0:
            expired = [
                key
                for key, (saved_at, _result) in self._cache.items()
                if now - saved_at > self.cache_ttl_seconds
            ]
            for key in expired:
                self._cache.pop(key, None)
        overflow = len(self._cache) - self.cache_max_entries
        if overflow <= 0:
            return
        oldest = sorted(self._cache.items(), key=lambda item: item[1][0])[:overflow]
        for key, _entry in oldest:
            self._cache.pop(key, None)

    async def _lookup_once(self, *, client: httpx.AsyncClient, query: str, count: int, freshness: str | None = None) -> WebSearchResult:
        payload: dict[str, Any] = {
            "query": query,
            "summary": self.summary_enabled,
            "count": count,
        }
        if freshness:
            payload["freshness"] = freshness
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = await client.post(self.endpoint, json=payload, headers=headers)
        response.raise_for_status()
        body = response.json()
        return _parse_bocha_response(query=query, body=body, resource_package_id=self.resource_package_id)


def format_web_search_system_prompt(result: WebSearchResult) -> str:
    lines = ["实时工具结果：Bocha 联网搜索\n", f"- 当前日期：{date.today().isoformat()}\n", f"- 查询：{result.query}\n"]
    for index, item in enumerate(result.results[:5], start=1):
        text = item.summary or item.snippet
        lines.append(f"- 结果 {index}：{item.title}\n")
        if item.date:
            lines.append(f"  日期：{item.date}\n")
        if text:
            lines.append(f"  摘要：{text[:260]}\n")
        if item.url:
            lines.append(f"  链接：{item.url}\n")
    lines.append("回答联网搜索问题时必须基于以上搜索结果；当前已经执行过搜索，不要再建议用户自行搜索。")
    lines.append("回答今年、最新、赛事、新闻等时效性问题时，必须结合当前日期和以上搜索结果。")
    lines.append("只有当搜索结果明确给出冠军、比分、决赛结果、赛程状态等关键信息时，才可以确认这些结论；如果结果不足或互相冲突，请明确说“我目前搜到的结果不足以确认”，不要按当前日期、旧赛程或常识倒推。\n")
    lines.append("赛事问题尤其注意：如果当前日期已经晚于搜索结果里提到的开赛、决赛或赛程日期，而搜索结果没有明确最终赛果，不要说“还没开始”“还没踢完”“将举行”；应说明这些搜索结果可能是赛前或过期信息，目前无法确认最终结果。\n")
    return "".join(lines)


def _parse_bocha_response(*, query: str, body: Any, resource_package_id: str) -> WebSearchResult:
    if not isinstance(body, dict):
        raise ValueError("Bocha response must be a JSON object")
    data = body.get("data") if isinstance(body.get("data"), dict) else body
    web_pages = data.get("webPages") if isinstance(data, dict) else None
    raw_items: Any = None
    if isinstance(web_pages, dict):
        raw_items = web_pages.get("value")
    if raw_items is None and isinstance(data, dict):
        raw_items = data.get("results") or data.get("items") or data.get("value")
    if not isinstance(raw_items, list):
        raw_items = []

    results: list[WebSearchItem] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("name") or item.get("title") or "").strip()
        url = str(item.get("url") or item.get("link") or "").strip()
        if not title and not url:
            continue
        results.append(
            WebSearchItem(
                title=title or url,
                url=url,
                snippet=str(item.get("snippet") or item.get("description") or "").strip(),
                summary=str(item.get("summary") or "").strip(),
                site_name=str(item.get("siteName") or item.get("site_name") or "").strip(),
                date=str(item.get("dateLastCrawled") or item.get("date") or "").strip(),
            )
        )
    if not results:
        raise ValueError("Bocha web search returned no results")
    return WebSearchResult(query=query, results=results, resource_package_id=resource_package_id)


def _expanded_world_cup_result_queries(query: str) -> list[str]:
    normalized = query.lower()
    if "世界杯" not in query and "world cup" not in normalized:
        return []
    result_markers = (
        "冠军",
        "夺冠",
        "决赛",
        "比分",
        "赛果",
        "结果",
        "winner",
        "champion",
        "final",
        "score",
        "result",
    )
    if not any(marker in normalized or marker in query for marker in result_markers):
        return []
    year_match = re.search(r"\b(20\d{2})\b", query)
    year = year_match.group(1) if year_match else str(date.today().year)
    return [
        f"site:fifa.com {year} FIFA World Cup final winner score",
        f"{year} FIFA World Cup final winner score result",
        f"{year} World Cup winner final score",
        f"{year} 世界杯 决赛 比分 冠军 夺冠",
    ]


def _unique_queries(queries: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for query in queries:
        key = re.sub(r"\s+", " ", query).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(query.strip())
    return unique


def _dedupe_search_items(items: list[WebSearchItem]) -> list[WebSearchItem]:
    deduped: list[WebSearchItem] = []
    seen: set[str] = set()
    for item in items:
        key = (item.url or item.title).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _normalize_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    normalized = re.sub(r"[？?。！!，,；;：:、]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _clean_search_query(text: str) -> str | None:
    cleaned = SEARCH_NOISE_PATTERN.sub(" ", text)
    cleaned = re.sub(r"(?:吗|呢|啊|吧|呀|么)$", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -_，,。？?：:")
    return cleaned[:120] if cleaned else None


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return float(raw)


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return int(raw)
