"""Free weather lookup tool with multiple provider backends."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WTTR_URL_TEMPLATE = "https://wttr.in/{query}"
CMA_WEATHER_URL = "https://weather.cma.cn/api/weather/view"
AMAP_WEATHER_URL = "https://restapi.amap.com/v3/weather/weatherInfo"
DEFAULT_TIMEOUT_SECONDS = 3.0
DEFAULT_CACHE_TTL_SECONDS = 300.0
DEFAULT_CITY = "北京"
DEFAULT_PROVIDER = "amap"

WEATHER_MARKERS = (
    "天气",
    "气温",
    "温度",
    "冷不冷",
    "热不热",
    "下雨",
    "降雨",
    "会不会雨",
    "要不要带伞",
    "穿什么",
)
KNOWN_CITIES = (
    "北京",
    "上海",
    "广州",
    "深圳",
    "杭州",
    "南京",
    "苏州",
    "成都",
    "重庆",
    "武汉",
    "西安",
    "天津",
    "青岛",
    "厦门",
    "长沙",
    "郑州",
    "济南",
    "沈阳",
    "大连",
    "哈尔滨",
    "黑龙江省",
    "黑龙江",
    "长春",
    "昆明",
    "贵阳",
    "南宁",
    "福州",
    "合肥",
    "南昌",
    "太原",
    "石家庄",
    "呼和浩特",
    "兰州",
    "银川",
    "西宁",
    "乌鲁木齐",
    "拉萨",
    "海口",
    "三亚",
    "香港",
    "澳门",
    "台北",
    "东京",
    "首尔",
    "伦敦",
    "巴黎",
    "纽约",
    "洛杉矶",
    "旧金山",
    "新加坡",
)
CITY_COORDINATES: dict[str, dict[str, str | float]] = {
    "北京": {"name": "北京", "country": "中国", "latitude": 39.9075, "longitude": 116.39723},
    "上海": {"name": "上海", "country": "中国", "latitude": 31.22222, "longitude": 121.45806},
    "广州": {"name": "广州", "country": "中国", "latitude": 23.11667, "longitude": 113.25},
    "深圳": {"name": "深圳", "country": "中国", "latitude": 22.54554, "longitude": 114.0683},
    "杭州": {"name": "杭州", "country": "中国", "latitude": 30.29365, "longitude": 120.16142},
    "南京": {"name": "南京", "country": "中国", "latitude": 32.06167, "longitude": 118.77778},
    "成都": {"name": "成都", "country": "中国", "latitude": 30.66667, "longitude": 104.06667},
    "重庆": {"name": "重庆", "country": "中国", "latitude": 29.56278, "longitude": 106.55278},
    "武汉": {"name": "武汉", "country": "中国", "latitude": 30.58333, "longitude": 114.26667},
    "西安": {"name": "西安", "country": "中国", "latitude": 34.25833, "longitude": 108.92861},
    "天津": {"name": "天津", "country": "中国", "latitude": 39.14222, "longitude": 117.17667},
    "哈尔滨": {"name": "哈尔滨", "country": "中国", "latitude": 45.8038, "longitude": 126.5349},
    "harbin": {"name": "Harbin", "country": "China", "latitude": 45.8038, "longitude": 126.5349},
    "Harbin": {"name": "Harbin", "country": "China", "latitude": 45.8038, "longitude": 126.5349},
    "黑龙江": {"name": "黑龙江（哈尔滨代表点）", "country": "中国", "latitude": 45.8038, "longitude": 126.5349},
    "黑龙江省": {"name": "黑龙江（哈尔滨代表点）", "country": "中国", "latitude": 45.8038, "longitude": 126.5349},
}

WEATHER_CODE_TEXT = {
    0: "晴",
    1: "大部晴朗",
    2: "局部多云",
    3: "阴",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "中等毛毛雨",
    55: "大毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "小阵雨",
    81: "中等阵雨",
    82: "强阵雨",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴大冰雹",
}
CITY_ALIASES = {
    "北京": "Beijing",
    "上海": "Shanghai",
    "广州": "Guangzhou",
    "深圳": "Shenzhen",
    "杭州": "Hangzhou",
    "南京": "Nanjing",
    "苏州": "Suzhou",
    "成都": "Chengdu",
    "重庆": "Chongqing",
    "武汉": "Wuhan",
    "西安": "Xi'an",
    "天津": "Tianjin",
    "青岛": "Qingdao",
    "厦门": "Xiamen",
    "长沙": "Changsha",
    "郑州": "Zhengzhou",
    "济南": "Jinan",
    "沈阳": "Shenyang",
    "大连": "Dalian",
    "哈尔滨": "Harbin",
    "harbin": "Harbin",
    "Harbin": "Harbin",
    "昆明": "Kunming",
    "黑龙江": "Harbin",
    "黑龙江省": "Harbin",
    "香港": "Hong Kong",
    "澳门": "Macau",
    "台北": "Taipei",
}
CMA_STATION_IDS = {
    "北京": "54511",
    "上海": "58362",
    "广州": "59287",
    "深圳": "59493",
    "杭州": "58457",
    "南京": "58238",
    "成都": "56294",
    "重庆": "57516",
    "武汉": "57494",
    "西安": "57036",
    "天津": "54527",
    "哈尔滨": "50953",
    "harbin": "50953",
    "Harbin": "50953",
    "黑龙江": "50953",
    "黑龙江省": "50953",
}
AMAP_AD_CODES = {
    "北京": "110000",
    "上海": "310000",
    "广州": "440100",
    "深圳": "440300",
    "杭州": "330100",
    "南京": "320100",
    "苏州": "320500",
    "成都": "510100",
    "重庆": "500000",
    "武汉": "420100",
    "西安": "610100",
    "天津": "120000",
    "青岛": "370200",
    "厦门": "350200",
    "长沙": "430100",
    "郑州": "410100",
    "济南": "370100",
    "沈阳": "210100",
    "大连": "210200",
    "哈尔滨": "230100",
    "harbin": "230100",
    "Harbin": "230100",
    "黑龙江": "230100",
    "黑龙江省": "230100",
    "长春": "220100",
    "昆明": "530100",
    "贵阳": "520100",
    "南宁": "450100",
    "福州": "350100",
    "合肥": "340100",
    "南昌": "360100",
    "太原": "140100",
    "石家庄": "130100",
    "呼和浩特": "150100",
    "兰州": "620100",
    "银川": "640100",
    "西宁": "630100",
    "乌鲁木齐": "650100",
    "拉萨": "540100",
    "海口": "460100",
    "三亚": "460200",
    "香港": "810000",
    "澳门": "820000",
    "台北": "710000",
}
WTTR_CONDITION_TEXT = {
    "Sunny": "晴",
    "Clear": "晴",
    "Partly Cloudy": "局部多云",
    "Partly cloudy": "局部多云",
    "Cloudy": "多云",
    "Overcast": "阴",
    "Mist": "薄雾",
    "Fog": "雾",
    "Smoky haze": "霾",
    "Patchy rain nearby": "附近有零星降雨",
    "Light rain": "小雨",
    "Moderate rain": "中雨",
    "Heavy rain": "大雨",
    "Light drizzle": "小毛毛雨",
    "Thundery outbreaks possible": "可能有雷阵雨",
}


@dataclass(frozen=True)
class WeatherToolResult:
    """Normalized weather result for prompt injection and API responses."""

    city: str
    country: str
    latitude: float
    longitude: float
    timezone: str
    day: str
    date: str
    condition: str
    temperature_c: float | None
    apparent_temperature_c: float | None
    humidity_percent: int | None
    precipitation_mm: float | None
    wind_speed_kmh: float | None
    wind_direction_degree: int | None
    temp_max_c: float | None
    temp_min_c: float | None
    precipitation_probability_max_percent: int | None
    source: str = "open-meteo"

    def to_dict(self) -> dict[str, Any]:
        return {
            "city": self.city,
            "country": self.country,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": self.timezone,
            "day": self.day,
            "date": self.date,
            "condition": self.condition,
            "temperature_c": self.temperature_c,
            "apparent_temperature_c": self.apparent_temperature_c,
            "humidity_percent": self.humidity_percent,
            "precipitation_mm": self.precipitation_mm,
            "wind_speed_kmh": self.wind_speed_kmh,
            "wind_direction_degree": self.wind_direction_degree,
            "temp_max_c": self.temp_max_c,
            "temp_min_c": self.temp_min_c,
            "precipitation_probability_max_percent": self.precipitation_probability_max_percent,
            "source": self.source,
        }


class OpenMeteoWeatherTool:
    """Lookup weather without an API key using Open-Meteo."""

    def __init__(self, *, timeout_seconds: float | None = None) -> None:
        self.timeout_seconds = timeout_seconds or _read_float_env("WEATHER_TOOL_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
        self.cache_ttl_seconds = _read_float_env("WEATHER_CACHE_TTL_SECONDS", DEFAULT_CACHE_TTL_SECONDS)
        self.max_stale_seconds = _read_float_env("WEATHER_STALE_CACHE_MAX_AGE_SECONDS", 6 * 3600.0)
        self.default_city = os.getenv("WEATHER_DEFAULT_CITY", DEFAULT_CITY).strip() or DEFAULT_CITY
        self.provider = os.getenv("WEATHER_PROVIDER", DEFAULT_PROVIDER).strip().lower() or DEFAULT_PROVIDER
        self.persistent_cache_enabled = _read_bool_env("WEATHER_PERSISTENT_CACHE_ENABLED", True)
        self.return_stale_while_refreshing = _read_bool_env("WEATHER_RETURN_STALE_WHILE_REFRESHING", True)
        self.cache_file = Path(os.getenv("WEATHER_CACHE_FILE", ".runtime/weather_cache.json")).expanduser()
        self._cache: dict[tuple[str, str, str], tuple[float, WeatherToolResult]] = {}
        self._refreshing: set[tuple[str, str, str]] = set()
        self._load_persistent_cache()

    def extract_query(self, text: str) -> tuple[str, str] | None:
        if not is_weather_query(text):
            return None
        return extract_city(text, default_city=self.default_city), extract_day(text)

    async def lookup_from_text(self, text: str) -> WeatherToolResult | None:
        extracted = self.extract_query(text)
        if extracted is None:
            return None
        city, day = extracted
        return await self.lookup(city=city, day=day)

    async def lookup(self, *, city: str, day: str = "today") -> WeatherToolResult:
        city = city.strip() or self.default_city
        day = normalize_day(day)
        legacy_cache_key = (self.provider, city, day)
        cache_key = (self.provider, _canonical_city_key(city), day)
        now = time.time()
        cached = self._cache.get(legacy_cache_key) or self._cache.get(cache_key)
        if cached and not _cache_matches_provider(self.provider, cached[1]):
            cached = None
        if cached:
            age = now - cached[0]
            if self.cache_ttl_seconds > 0 and age <= self.cache_ttl_seconds:
                return cached[1]
            if self.return_stale_while_refreshing and 0 <= age <= self.max_stale_seconds:
                self._refresh_cache_in_background(city=city, day=day, cache_key=cache_key)
                return _stale_result(cached[1])
        stale_result = cached[1] if cached else None
        try:
            cache_key, result = await self._lookup_uncached(city=city, day=day, cache_key=cache_key)
        except Exception as exc:
            if _read_bool_env("WEATHER_STALE_CACHE_FALLBACK_ENABLED", True):
                stale_result = stale_result or self._find_any_cached(city=city, day=day)
                if stale_result is not None:
                    return _stale_result(stale_result)
            if _read_bool_env("WEATHER_UNAVAILABLE_FALLBACK_ENABLED", True):
                return _unavailable_result(city=city, day=day, error=str(exc)[:160])
            raise
        self._store_cache(cache_key, result)
        if legacy_cache_key != cache_key and legacy_cache_key[0] == cache_key[0]:
            self._cache[legacy_cache_key] = self._cache[cache_key]
            self._save_persistent_cache()
        return result

    async def _lookup_uncached(
        self,
        *,
        city: str,
        day: str,
        cache_key: tuple[str, str, str],
    ) -> tuple[tuple[str, str, str], WeatherToolResult]:
        if self.provider in {"auto", "race"}:
            return cache_key, await self._lookup_first_available(city=city, day=day)
        if self.provider in {"amap", "gaode"}:
            try:
                return cache_key, await self._lookup_amap(city=city, day=day)
            except Exception:
                if _read_bool_env("WEATHER_AMAP_FALLBACK_ENABLED", True):
                    return await self._lookup_fallback_after_amap(city=city, day=day)
                raise
        if self.provider == "cma":
            try:
                return cache_key, await self._lookup_cma(city=city, day=day)
            except Exception:
                if _read_bool_env("WEATHER_CMA_FALLBACK_ENABLED", True):
                    return ("wttr", _canonical_city_key(city), day), await self._lookup_wttr(city=city, day=day)
                raise
        if self.provider == "wttr":
            try:
                return cache_key, await self._lookup_wttr(city=city, day=day)
            except Exception:
                if _read_bool_env("WEATHER_WTTR_FALLBACK_ENABLED", True):
                    return ("open-meteo", city, day), await self._lookup_open_meteo(city=city, day=day)
                raise
        if self.provider in {"open-meteo", "openmeteo"}:
            return cache_key, await self._lookup_open_meteo(city=city, day=day)
        raise ValueError(f"Unsupported WEATHER_PROVIDER: {self.provider}")

    async def _lookup_fallback_after_amap(self, *, city: str, day: str) -> tuple[tuple[str, str, str], WeatherToolResult]:
        if _cma_station_id(city) is not None:
            try:
                return ("cma", _canonical_city_key(city), day), await self._lookup_cma(city=city, day=day)
            except Exception:
                pass
        return ("wttr", _canonical_city_key(city), day), await self._lookup_wttr(city=city, day=day)

    def _store_cache(self, cache_key: tuple[str, str, str], result: WeatherToolResult) -> None:
        if self.cache_ttl_seconds <= 0:
            return
        self._cache[cache_key] = (time.time(), result)
        self._save_persistent_cache()

    def _refresh_cache_in_background(self, *, city: str, day: str, cache_key: tuple[str, str, str]) -> None:
        if cache_key in self._refreshing:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._refreshing.add(cache_key)
        task = loop.create_task(self._refresh_cache(city=city, day=day, cache_key=cache_key))
        task.add_done_callback(lambda _: self._refreshing.discard(cache_key))

    async def _refresh_cache(self, *, city: str, day: str, cache_key: tuple[str, str, str]) -> None:
        try:
            refreshed_key, result = await self._lookup_uncached(city=city, day=day, cache_key=cache_key)
            self._store_cache(refreshed_key, result)
        except Exception:
            return

    def _load_persistent_cache(self) -> None:
        if not self.persistent_cache_enabled:
            return
        try:
            if not self.cache_file.exists():
                return
            body = json.loads(self.cache_file.read_text(encoding="utf-8"))
            items = body.get("items") if isinstance(body, dict) else []
            if not isinstance(items, list):
                return
            for item in items:
                if not isinstance(item, dict):
                    continue
                raw_key = item.get("key")
                data = item.get("result")
                if not isinstance(raw_key, list) or len(raw_key) != 3 or not isinstance(data, dict):
                    continue
                key = (str(raw_key[0]), str(raw_key[1]), str(raw_key[2]))
                self._cache[key] = (float(item.get("saved_at") or 0), WeatherToolResult(**data))
        except Exception:
            self._cache.clear()

    def _save_persistent_cache(self) -> None:
        if not self.persistent_cache_enabled:
            return
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            items = [
                {"key": list(key), "saved_at": saved_at, "result": result.to_dict()}
                for key, (saved_at, result) in sorted(self._cache.items(), key=lambda item: item[1][0], reverse=True)[:200]
            ]
            self.cache_file.write_text(json.dumps({"items": items}, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            return

    def _find_any_cached(self, *, city: str, day: str) -> WeatherToolResult | None:
        city_keys = {city, _canonical_city_key(city)}
        alias = CITY_ALIASES.get(city)
        if alias:
            city_keys.update({alias, alias.lower()})
        for (_provider, cached_city, cached_day), (_saved_at, result) in sorted(
            self._cache.items(),
            key=lambda item: item[1][0],
            reverse=True,
        ):
            if cached_day != day:
                continue
            if cached_city in city_keys or _canonical_city_key(cached_city) in city_keys:
                return result
        return None

    async def _lookup_first_available(self, *, city: str, day: str) -> WeatherToolResult:
        tasks = []
        if _amap_api_key() and _amap_city_code(city) is not None:
            tasks.append(asyncio.create_task(self._lookup_amap(city=city, day=day)))
        if _cma_station_id(city) is not None:
            tasks.append(asyncio.create_task(self._lookup_cma(city=city, day=day)))
        tasks.append(asyncio.create_task(self._lookup_wttr(city=city, day=day)))
        if _read_bool_env("WEATHER_OPEN_METEO_RACE_ENABLED", False):
            tasks.append(asyncio.create_task(self._lookup_open_meteo(city=city, day=day)))
        errors: list[str] = []
        try:
            for task in asyncio.as_completed(tasks):
                try:
                    return await task
                except Exception as exc:
                    errors.append(str(exc)[:160])
            raise RuntimeError("All weather providers failed: " + "; ".join(errors))
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def _lookup_open_meteo(self, *, city: str, day: str) -> WeatherToolResult:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            location = CITY_COORDINATES.get(city)
            if location is None:
                geo_response = await client.get(
                    GEOCODING_URL,
                    params={"name": city, "count": 1, "language": "zh", "format": "json"},
                )
                geo_response.raise_for_status()
                location = _first_location(geo_response.json(), city)

            forecast_response = await client.get(
                FORECAST_URL,
                params={
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    "current": ",".join(
                        [
                            "temperature_2m",
                            "relative_humidity_2m",
                            "apparent_temperature",
                            "precipitation",
                            "weather_code",
                            "wind_speed_10m",
                            "wind_direction_10m",
                        ]
                    ),
                    "daily": ",".join(
                        [
                            "weather_code",
                            "temperature_2m_max",
                            "temperature_2m_min",
                            "precipitation_probability_max",
                        ]
                    ),
                    "forecast_days": 3,
                    "timezone": "auto",
                },
            )
            forecast_response.raise_for_status()
            forecast = forecast_response.json()
        return _to_result(city=city, day=day, location=location, forecast=forecast)

    async def _lookup_amap(self, *, city: str, day: str) -> WeatherToolResult:
        api_key = _amap_api_key()
        if not api_key:
            raise ValueError("AMAP_WEATHER_API_KEY is not configured")
        city_code = _amap_city_code(city)
        if city_code is None:
            raise ValueError(f"AMap weather adcode not configured for city: {city}")
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(
                AMAP_WEATHER_URL,
                params={
                    "key": api_key,
                    "city": city_code,
                    "extensions": "all",
                    "output": "JSON",
                },
                headers={"User-Agent": "navi-weather-tool/1.0"},
            )
            response.raise_for_status()
            body = response.json()
        return _to_amap_result(city=city, day=day, body=body)

    async def _lookup_cma(self, *, city: str, day: str) -> WeatherToolResult:
        station_id = _cma_station_id(city)
        if station_id is None:
            raise ValueError(f"CMA weather station not configured for city: {city}")
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(
                CMA_WEATHER_URL,
                params={"stationid": station_id},
                headers={"User-Agent": "navi-weather-tool/1.0"},
            )
            response.raise_for_status()
            body = response.json()
        return _to_cma_result(city=city, day=day, body=body)

    async def _lookup_wttr(self, *, city: str, day: str) -> WeatherToolResult:
        query = CITY_ALIASES.get(city, city)
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(
                WTTR_URL_TEMPLATE.format(query=quote(query)),
                params={"format": "j1", "lang": "zh"},
                headers={"User-Agent": "navi-weather-tool/1.0"},
            )
            response.raise_for_status()
            body = response.json()
        return _to_wttr_result(city=city, day=day, body=body)


def is_weather_query(text: str) -> bool:
    normalized = text.strip()
    return bool(normalized) and any(marker in normalized for marker in WEATHER_MARKERS)


def extract_city(text: str, *, default_city: str = DEFAULT_CITY) -> str:
    for city in sorted(KNOWN_CITIES, key=len, reverse=True):
        if city in text:
            return city
    patterns = [
        r"(?:查一下|查询|看看|告诉我)?([\u4e00-\u9fffA-Za-z\s]{2,40}?)(?:今天|明天|后天|现在|当前)?(?:的)?(?:天气|气温|温度)",
        r"(?:今天|明天|后天|现在|当前)?([\u4e00-\u9fffA-Za-z\s]{2,40}?)(?:今天|明天|后天|现在|当前)?(?:会不会|会|是否|要不要)?(?:下雨|降雨)",
        r"(?:天气|气温|温度).*?(?:在|查|查询|看看)([\u4e00-\u9fffA-Za-z\s]{2,40})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidate = _clean_city_candidate(match.group(1))
            if candidate:
                return candidate
    return default_city


def extract_day(text: str) -> str:
    if "后天" in text:
        return "day_after_tomorrow"
    if "明天" in text:
        return "tomorrow"
    return "today"


def normalize_day(day: str) -> str:
    raw = (day or "today").strip().lower()
    aliases = {
        "今天": "today",
        "today": "today",
        "now": "today",
        "current": "today",
        "明天": "tomorrow",
        "tomorrow": "tomorrow",
        "后天": "day_after_tomorrow",
        "day_after_tomorrow": "day_after_tomorrow",
    }
    return aliases.get(raw, "today")


def format_weather_system_prompt(result: WeatherToolResult) -> str:
    data = result.to_dict()
    lines = [
        "实时工具结果：天气查询\n"
        f"- 来源：{result.source}\n"
        f"- 地点：{result.city}，{result.country}\n"
        f"- 日期：{result.date}（{_day_label(result.day)}）\n"
        f"- 天气：{result.condition}\n"
    ]
    if result.day == "today":
        lines.extend(
            [
                f"- 当前温度：{_value(data['temperature_c'], '°C')}，体感：{_value(data['apparent_temperature_c'], '°C')}\n",
                f"- 今日范围：{_value(data['temp_min_c'], '°C')} 到 {_value(data['temp_max_c'], '°C')}\n",
                f"- 湿度：{_value(data['humidity_percent'], '%')}，当前降水：{_value(data['precipitation_mm'], 'mm')}\n",
                f"- 今日最大降水概率：{_value(data['precipitation_probability_max_percent'], '%')}\n",
                f"- 风速：{_value(data['wind_speed_kmh'], 'km/h')}，风向：{_value(data['wind_direction_degree'], '°')}\n",
            ]
        )
    else:
        lines.extend(
            [
                f"- 预报温度范围：{_value(data['temp_min_c'], '°C')} 到 {_value(data['temp_max_c'], '°C')}\n",
                f"- 最大降水概率：{_value(data['precipitation_probability_max_percent'], '%')}\n",
            ]
        )
    lines.append("回答天气问题时必须基于以上实时工具结果；如果学生问是否带伞或穿衣，请给一句实用建议。")
    return "".join(lines)


def _first_location(body: dict[str, Any], city: str) -> dict[str, Any]:
    results = body.get("results") or []
    if not results:
        raise ValueError(f"未找到城市：{city}")
    location = results[0]
    return {
        "name": str(location.get("name") or city),
        "country": str(location.get("country") or ""),
        "latitude": float(location["latitude"]),
        "longitude": float(location["longitude"]),
    }


def _to_result(*, city: str, day: str, location: dict[str, Any], forecast: dict[str, Any]) -> WeatherToolResult:
    current = forecast.get("current") or {}
    daily = forecast.get("daily") or {}
    index = {"today": 0, "tomorrow": 1, "day_after_tomorrow": 2}.get(day, 0)

    daily_time = _list_get(daily.get("time"), index) or str(current.get("time") or "")
    if day == "today":
        weather_code = current.get("weather_code")
        if weather_code is None:
            weather_code = _list_get(daily.get("weather_code"), index)
    else:
        weather_code = _list_get(daily.get("weather_code"), index)
    condition = WEATHER_CODE_TEXT.get(int(weather_code), f"天气代码 {weather_code}") if weather_code is not None else "未知"

    return WeatherToolResult(
        city=str(location.get("name") or city),
        country=str(location.get("country") or ""),
        latitude=float(location["latitude"]),
        longitude=float(location["longitude"]),
        timezone=str(forecast.get("timezone") or ""),
        day=day,
        date=str(daily_time),
        condition=condition,
        temperature_c=_number_or_none(current.get("temperature_2m")) if day == "today" else None,
        apparent_temperature_c=_number_or_none(current.get("apparent_temperature")) if day == "today" else None,
        humidity_percent=_int_or_none(current.get("relative_humidity_2m")) if day == "today" else None,
        precipitation_mm=_number_or_none(current.get("precipitation")) if day == "today" else None,
        wind_speed_kmh=_number_or_none(current.get("wind_speed_10m")) if day == "today" else None,
        wind_direction_degree=_int_or_none(current.get("wind_direction_10m")) if day == "today" else None,
        temp_max_c=_number_or_none(_list_get(daily.get("temperature_2m_max"), index)),
        temp_min_c=_number_or_none(_list_get(daily.get("temperature_2m_min"), index)),
        precipitation_probability_max_percent=_int_or_none(_list_get(daily.get("precipitation_probability_max"), index)),
    )


def _to_amap_result(*, city: str, day: str, body: dict[str, Any]) -> WeatherToolResult:
    if str(body.get("status") or "") != "1":
        info = body.get("info") or body.get("infocode") or "AMap weather response error"
        raise ValueError(str(info))
    forecasts = body.get("forecasts")
    if not isinstance(forecasts, list) or not forecasts:
        raise ValueError("AMap weather response missing forecasts")
    forecast = forecasts[0] if isinstance(forecasts[0], dict) else {}
    casts = forecast.get("casts") or []
    if not isinstance(casts, list):
        casts = []
    index = {"today": 0, "tomorrow": 1, "day_after_tomorrow": 2}.get(day, 0)
    cast = _list_get(casts, index) or {}
    if not isinstance(cast, dict):
        cast = {}

    day_weather = str(cast.get("dayweather") or "").strip()
    night_weather = str(cast.get("nightweather") or "").strip()
    if day_weather and night_weather and day_weather != night_weather:
        condition = f"{day_weather}转{night_weather}"
    else:
        condition = day_weather or night_weather or "未知"

    location = CITY_COORDINATES.get(city) or CITY_COORDINATES.get(str(forecast.get("city") or "")) or {}
    return WeatherToolResult(
        city=city or str(forecast.get("city") or ""),
        country="中国",
        latitude=float(location.get("latitude") or 0.0),
        longitude=float(location.get("longitude") or 0.0),
        timezone="UTC+8",
        day=day,
        date=str(cast.get("date") or ""),
        condition=condition,
        temperature_c=None,
        apparent_temperature_c=None,
        humidity_percent=None,
        precipitation_mm=None,
        wind_speed_kmh=None,
        wind_direction_degree=None,
        temp_max_c=_number_or_none(cast.get("daytemp")),
        temp_min_c=_number_or_none(cast.get("nighttemp")),
        precipitation_probability_max_percent=None,
        source="amap",
    )


def _to_cma_result(*, city: str, day: str, body: dict[str, Any]) -> WeatherToolResult:
    if int(body.get("code") or 0) != 0:
        raise ValueError(str(body.get("msg") or "CMA weather response error"))
    data = body.get("data")
    if not isinstance(data, dict):
        raise ValueError("CMA weather response missing data")
    location = data.get("location") or {}
    daily_items = data.get("daily") or []
    now = data.get("now") or {}
    index = {"today": 0, "tomorrow": 1, "day_after_tomorrow": 2}.get(day, 0)
    daily = _list_get(daily_items, index) or {}
    if not isinstance(daily, dict):
        daily = {}

    day_text = str(daily.get("dayText") or "").strip()
    night_text = str(daily.get("nightText") or "").strip()
    if day_text and night_text and day_text != night_text:
        condition = f"{day_text}转{night_text}"
    else:
        condition = day_text or night_text or "未知"

    path = str(location.get("path") or "")
    country = path.split(",", 1)[0].strip() if path else "中国"
    wind_speed = _number_or_none(now.get("windSpeed")) if day == "today" else None
    return WeatherToolResult(
        city=city or str(location.get("name") or ""),
        country=country,
        latitude=float(location.get("latitude") or 0.0),
        longitude=float(location.get("longitude") or 0.0),
        timezone=f"UTC+{location.get('timezone')}" if location.get("timezone") is not None else "",
        day=day,
        date=str(daily.get("date") or "").replace("/", "-"),
        condition=condition,
        temperature_c=_number_or_none(now.get("temperature")) if day == "today" else None,
        apparent_temperature_c=_number_or_none(now.get("feelst")) if day == "today" else None,
        humidity_percent=_int_or_none(now.get("humidity")) if day == "today" else None,
        precipitation_mm=_number_or_none(now.get("precipitation")) if day == "today" else None,
        wind_speed_kmh=round(wind_speed * 3.6, 1) if wind_speed is not None and day == "today" else None,
        wind_direction_degree=_int_or_none(now.get("windDirectionDegree")) if day == "today" else None,
        temp_max_c=_number_or_none(daily.get("high")),
        temp_min_c=_number_or_none(daily.get("low")),
        precipitation_probability_max_percent=None,
        source="cma",
    )


def _to_wttr_result(*, city: str, day: str, body: dict[str, Any]) -> WeatherToolResult:
    index = {"today": 0, "tomorrow": 1, "day_after_tomorrow": 2}.get(day, 0)
    current = _list_get(body.get("current_condition"), 0) or {}
    area = _list_get(body.get("nearest_area"), 0) or {}
    daily = _list_get(body.get("weather"), index) or {}

    current_desc = _extract_nested_value(current.get("weatherDesc"))
    daily_desc = _extract_nested_value((_list_get(daily.get("hourly"), 4) or {}).get("weatherDesc"))
    raw_condition = current_desc if day == "today" else daily_desc
    condition = WTTR_CONDITION_TEXT.get(str(raw_condition).strip(), str(raw_condition or "未知").strip())

    area_name = _extract_nested_value(area.get("areaName")) or city
    country = _extract_nested_value(area.get("country")) or ""
    return WeatherToolResult(
        city=city or str(area_name),
        country=str(country),
        latitude=float(area.get("latitude") or 0.0),
        longitude=float(area.get("longitude") or 0.0),
        timezone="",
        day=day,
        date=str(daily.get("date") or ""),
        condition=condition,
        temperature_c=_number_or_none(current.get("temp_C")) if day == "today" else None,
        apparent_temperature_c=_number_or_none(current.get("FeelsLikeC")) if day == "today" else None,
        humidity_percent=_int_or_none(current.get("humidity")) if day == "today" else None,
        precipitation_mm=_number_or_none(current.get("precipMM")) if day == "today" else None,
        wind_speed_kmh=_number_or_none(current.get("windspeedKmph")) if day == "today" else None,
        wind_direction_degree=_int_or_none(current.get("winddirDegree")) if day == "today" else None,
        temp_max_c=_number_or_none(daily.get("maxtempC")),
        temp_min_c=_number_or_none(daily.get("mintempC")),
        precipitation_probability_max_percent=_max_hourly_chance_of_rain(daily),
        source="wttr",
    )


def _stale_result(result: WeatherToolResult) -> WeatherToolResult:
    source = result.source if result.source.endswith("-stale") else f"{result.source}-stale"
    return replace(result, source=source)


def _cache_matches_provider(provider: str, result: WeatherToolResult) -> bool:
    normalized = provider.strip().lower()
    source = result.source.removesuffix("-stale")
    if normalized in {"amap", "gaode"} and _amap_api_key():
        return source == "amap"
    if normalized == "cma":
        return source == "cma"
    if normalized == "wttr":
        return source == "wttr"
    if normalized in {"open-meteo", "openmeteo"}:
        return source == "open-meteo"
    return True


def _unavailable_result(*, city: str, day: str, error: str) -> WeatherToolResult:
    location = CITY_COORDINATES.get(city) or CITY_COORDINATES.get(_canonical_city_key(city)) or {}
    return WeatherToolResult(
        city=str(location.get("name") or city or DEFAULT_CITY),
        country=str(location.get("country") or ""),
        latitude=float(location.get("latitude") or 0.0),
        longitude=float(location.get("longitude") or 0.0),
        timezone="",
        day=day,
        date=time.strftime("%Y-%m-%d"),
        condition=f"实时天气暂不可用：{error}" if error else "实时天气暂不可用",
        temperature_c=None,
        apparent_temperature_c=None,
        humidity_percent=None,
        precipitation_mm=None,
        wind_speed_kmh=None,
        wind_direction_degree=None,
        temp_max_c=None,
        temp_min_c=None,
        precipitation_probability_max_percent=None,
        source="weather-unavailable",
    )


def _canonical_city_key(city: str) -> str:
    normalized = str(city or "").strip()
    alias = CITY_ALIASES.get(normalized, normalized).strip()
    return (alias or normalized).lower()


def _cma_station_id(city: str) -> str | None:
    normalized = str(city or "").strip()
    return CMA_STATION_IDS.get(normalized) or CMA_STATION_IDS.get(_canonical_city_key(normalized))


def _amap_city_code(city: str) -> str | None:
    normalized = str(city or "").strip()
    if re.fullmatch(r"\d{6}", normalized):
        return normalized
    return AMAP_AD_CODES.get(normalized) or AMAP_AD_CODES.get(_canonical_city_key(normalized))


def _amap_api_key() -> str:
    for name in ("AMAP_WEATHER_API_KEY", "AMAP_API_KEY", "GAODE_API_KEY"):
        value = os.getenv(name, "").strip()
        if value and not value.startswith("REPLACE_WITH"):
            return value
    return ""


def _clean_city_candidate(value: str) -> str:
    text = re.sub(r"(我想|请|帮我|一下|今天|明天|后天|现在|当前|的|会不会|是否|需要|要不要|带伞|穿什么)", "", value)
    text = re.sub(r"(?:会|吗|呢|吧|呀|么)$", "", text)
    text = re.sub(r"[，。！？,.!?;；:：]", "", text).strip()
    return text[:40]


def _day_label(day: str) -> str:
    return {"today": "今天", "tomorrow": "明天", "day_after_tomorrow": "后天"}.get(day, "今天")


def _value(value: Any, unit: str) -> str:
    if value is None:
        return "未知"
    return f"{value}{unit}"


def _list_get(value: Any, index: int) -> Any:
    if not isinstance(value, list) or index >= len(value):
        return None
    return value[index]


def _extract_nested_value(value: Any) -> str:
    item = _list_get(value, 0)
    if isinstance(item, dict):
        return str(item.get("value") or "")
    return str(item or "")


def _max_hourly_chance_of_rain(daily: dict[str, Any]) -> int | None:
    hourly = daily.get("hourly")
    if not isinstance(hourly, list):
        return None
    values: list[int] = []
    for item in hourly:
        if not isinstance(item, dict):
            continue
        value = _int_or_none(item.get("chanceofrain"))
        if value is not None:
            values.append(value)
    return max(values) if values else None


def _number_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 1)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(round(float(value)))


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return float(raw)


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}
