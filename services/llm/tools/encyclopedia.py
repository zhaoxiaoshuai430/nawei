"""Wikipedia encyclopedia lookup tool."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL_TEMPLATE = "https://{language}.wikipedia.org/w/api.php"
DEFAULT_TIMEOUT_SECONDS = 8.0
DEFAULT_CACHE_TTL_SECONDS = 3600.0
ENCYCLOPEDIA_MARKERS = (
    "百科",
    "介绍",
    "简介",
    "生平",
    "背景",
    "资料",
    "是谁",
    "是什么",
    "什么是",
    "维基",
    "wikipedia",
    "联网",
    "上网",
    "网上",
    "网络",
    "搜索",
    "搜搜",
    "搜一下",
    "百度",
    "谷歌",
    "google",
    "查找",
    "查资料",
    "查一下",
    "查一查",
    "查下",
    "查查",
    "查",
    "查询",
    "了解一下",
    "了解",
    "科普",
)
WEATHER_LIKE_MARKERS = ("天气", "气温", "温度", "下雨", "降雨", "带伞", "穿什么")
DICTIONARY_LIKE_MARKERS = ("单词", "英文", "英语", "怎么读", "definition", "define", "meaning", "什么意思")
TIME_LIKE_PATTERNS = (
    r"(?:今天|现在|当前).*(?:几号|星期几|日期|时间|几点)",
    r"(?:几号|星期几|几点了|现在几点)",
)
CALCULATOR_LIKE_MARKERS = ("计算", "算一下", "等于多少", "是多少", "平方根", "开方")
NON_ENCYCLOPEDIA_MARKERS = (
    "自我介绍",
    "介绍自己",
    "介绍一下自己",
    "介绍你自己",
    "介绍一下你自己",
    "你是谁",
    "你叫什么",
    "你的名字",
    "一句介绍",
    "简短介绍",
)
ENCYCLOPEDIA_PATTERNS = (
    r"^(?:请|麻烦|帮我|给我|你|我想)?(?:联网|上网|网上|网络|维基|百科)?(?:查一查|查一下|查下|查查|查询|查找|搜索|搜一下|搜搜|百度一下|百度|谷歌一下|谷歌|google|搜|查|找一下|找找|找|检索)(?:一下|下)?(?:关于|有关)?(.+?)(?:的)?(?:资料|信息|百科|介绍|简介)?$",
    r"^(?:请|麻烦|帮我|给我|你|我想)?(?:介绍一下|介绍|讲一下|讲讲|说一下|说说|了解一下|了解|科普一下|科普)(?:关于|有关)?(.+)$",
    r"^(?:什么是|啥是)(.+)$",
    r"^(.+?)(?:是(?:谁|什么)|叫什么)$",
    r"^(.+?)(?:的)?(?:百科|资料|介绍|简介|生平|履历|背景)$",
    r"^(.+?)(?:有(?:哪些|什么)|有什么)(?:成就|贡献|作品|代表作|经历)$",
)
QUERY_NOISE_PATTERN = re.compile(
    r"(?:请|麻烦|帮我|给我|你|详细|简单|一下|下|关于|有关|"
    r"百科|介绍一下|介绍|简介|讲一下|讲讲|说一下|说说|了解一下|了解|科普一下|科普|"
    r"联网|上网|网上|网络|搜索|搜一下|搜搜|百度一下|百度|谷歌一下|谷歌|google|查找|查资料|查一查|查一下|查下|查查|查询|检索|我想|维基|wikipedia|"
    r"是什么|什么是|是谁|叫什么|资料|信息)",
    flags=re.I,
)
DEFAULT_USER_AGENT = "XiaoweiStudentTutor/1.0 (local educational assistant)"


@dataclass(frozen=True)
class EncyclopediaResult:
    query: str
    title: str
    extract: str
    url: str | None
    language: str
    source: str = "wikipedia"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "title": self.title,
            "extract": self.extract,
            "url": self.url,
            "language": self.language,
            "source": self.source,
        }


class EncyclopediaTool:
    def __init__(self, *, timeout_seconds: float | None = None) -> None:
        self.timeout_seconds = timeout_seconds or _read_float_env("ENCYCLOPEDIA_TOOL_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
        self.cache_ttl_seconds = _read_float_env("ENCYCLOPEDIA_CACHE_TTL_SECONDS", DEFAULT_CACHE_TTL_SECONDS)
        self.default_language = os.getenv("ENCYCLOPEDIA_DEFAULT_LANGUAGE", "zh").strip() or "zh"
        self.default_variant = os.getenv("ENCYCLOPEDIA_DEFAULT_VARIANT", "zh-cn").strip() or "zh-cn"
        self.user_agent = os.getenv("ENCYCLOPEDIA_USER_AGENT", DEFAULT_USER_AGENT).strip() or DEFAULT_USER_AGENT
        self._cache: dict[tuple[str, str], tuple[float, EncyclopediaResult]] = {}

    def extract_query(self, text: str) -> str | None:
        normalized = _normalize_text(text)
        if not normalized:
            return None
        if any(marker in normalized for marker in WEATHER_LIKE_MARKERS):
            return None
        if any(marker.lower() in normalized.lower() for marker in DICTIONARY_LIKE_MARKERS):
            return None
        if any(re.search(pattern, normalized) for pattern in TIME_LIKE_PATTERNS):
            return None
        if any(marker in normalized for marker in CALCULATOR_LIKE_MARKERS) or re.search(r"\d\s*[\+\-\*/×÷^]\s*\d", normalized):
            return None
        if any(marker in normalized for marker in NON_ENCYCLOPEDIA_MARKERS):
            return None
        if not any(marker.lower() in normalized.lower() for marker in ENCYCLOPEDIA_MARKERS):
            return None

        for pattern in ENCYCLOPEDIA_PATTERNS:
            match = re.search(pattern, normalized, flags=re.I)
            if match:
                query = _clean_query(match.group(1))
                if query:
                    return query

        query = _clean_query(QUERY_NOISE_PATTERN.sub(" ", normalized))
        return query[:80] if query else None

    async def lookup_from_text(self, text: str) -> EncyclopediaResult | None:
        query = self.extract_query(text)
        if not query:
            return None
        return await self.lookup(query=query)

    async def lookup(self, *, query: str, language: str | None = None) -> EncyclopediaResult:
        language = (language or self.default_language).strip() or "zh"
        variant = self.default_variant if language.startswith("zh") else ""
        cache_key = (f"{language}:{variant}", query.strip().lower())
        cached = self._cache.get(cache_key)
        now = time.monotonic()
        if cached and self.cache_ttl_seconds > 0 and now - cached[0] <= self.cache_ttl_seconds:
            return cached[1]
        api_url = API_URL_TEMPLATE.format(language=language)
        headers = {"Accept": "application/json", "User-Agent": self.user_agent}
        variant_params = {"variant": variant, "converttitles": 1} if variant else {}
        search_body = await asyncio.to_thread(
            _fetch_json,
            api_url,
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 1,
                "format": "json",
                "utf8": 1,
                "origin": "*",
                **variant_params,
            },
            headers,
            self.timeout_seconds,
        )
        title = _first_search_title(search_body, query)
        page_body = await asyncio.to_thread(
            _fetch_json,
            api_url,
            {
                "action": "query",
                "prop": "extracts|info",
                "exintro": 1,
                "explaintext": 1,
                "inprop": "url",
                "redirects": 1,
                "titles": title,
                "format": "json",
                "utf8": 1,
                "origin": "*",
                **variant_params,
            },
            headers,
            self.timeout_seconds,
        )
        page = _first_page(page_body, title)
        result = EncyclopediaResult(
            query=query,
            title=str(page.get("title") or title),
            extract=str(page.get("extract") or ""),
            url=page.get("fullurl"),
            language=language,
        )
        if self.cache_ttl_seconds > 0:
            self._cache[cache_key] = (now, result)
        return result


def _fetch_json(url: str, params: dict[str, object], headers: dict[str, str], timeout_seconds: float) -> dict[str, Any]:
    request = Request(
        f"{url}?{urlencode(params)}",
        headers=headers,
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Encyclopedia upstream request failed: {exc}") from exc


def _normalize_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    normalized = re.sub(r"[？?。！!，,；;：:、]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _clean_query(query: str) -> str | None:
    cleaned = _normalize_text(query)
    cleaned = QUERY_NOISE_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"(?:吗|呢|啊|吧|呀|么)$", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -_，,。？?：:")
    if len(cleaned) < 2:
        return None
    return cleaned[:80]


def format_encyclopedia_system_prompt(result: EncyclopediaResult) -> str:
    return (
        "实时工具结果：百科查询\n"
        f"- 查询：{result.query}\n"
        f"- 标题：{result.title}\n"
        f"- 摘要：{result.extract[:600]}\n"
        f"- 来源：{result.url or result.source}\n"
        "回答百科事实问题时必须基于以上结果；如果结果和问题不匹配，要说明可能需要更具体的关键词。"
    )


def _first_search_title(body: dict[str, Any], query: str) -> str:
    results = ((body.get("query") or {}).get("search") or [])
    if not results:
        raise ValueError(f"未找到百科条目：{query}")
    return str(results[0]["title"])


def _first_page(body: dict[str, Any], title: str) -> dict[str, Any]:
    pages = ((body.get("query") or {}).get("pages") or {})
    for page in pages.values():
        if not isinstance(page, dict) or page.get("missing") is not None:
            continue
        if str(page.get("extract") or "").strip():
            return page
    raise ValueError(f"未找到百科摘要：{title}")


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return float(raw)
