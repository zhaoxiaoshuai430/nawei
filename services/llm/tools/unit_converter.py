"""Deterministic unit conversion tool."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


UNIT_ALIASES = {
    "m": "m",
    "米": "m",
    "meter": "m",
    "meters": "m",
    "km": "km",
    "公里": "km",
    "千米": "km",
    "cm": "cm",
    "厘米": "cm",
    "mm": "mm",
    "毫米": "mm",
    "kg": "kg",
    "千克": "kg",
    "公斤": "kg",
    "g": "g",
    "克": "g",
    "mg": "mg",
    "毫克": "mg",
    "s": "s",
    "秒": "s",
    "min": "min",
    "分钟": "min",
    "h": "h",
    "小时": "h",
    "km/h": "km/h",
    "公里每小时": "km/h",
    "m/s": "m/s",
    "米每秒": "m/s",
    "c": "c",
    "°c": "c",
    "摄氏度": "c",
    "f": "f",
    "°f": "f",
    "华氏度": "f",
}
LINEAR_FACTORS = {
    "m": ("length", 1.0),
    "km": ("length", 1000.0),
    "cm": ("length", 0.01),
    "mm": ("length", 0.001),
    "kg": ("mass", 1.0),
    "g": ("mass", 0.001),
    "mg": ("mass", 0.000001),
    "s": ("time", 1.0),
    "min": ("time", 60.0),
    "h": ("time", 3600.0),
    "m/s": ("speed", 1.0),
    "km/h": ("speed", 1000.0 / 3600.0),
}


@dataclass(frozen=True)
class UnitConversionResult:
    value: float
    from_unit: str
    to_unit: str
    result: float
    result_text: str
    source: str = "local-unit-converter"

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "from_unit": self.from_unit,
            "to_unit": self.to_unit,
            "result": self.result,
            "result_text": self.result_text,
            "source": self.source,
        }


class UnitConverterTool:
    def convert_from_text(self, text: str) -> UnitConversionResult | None:
        match = re.search(
            r"([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z/°\u4e00-\u9fff]+)\s*(?:等于|是多少|换成|转成|转换成|到)\s*([A-Za-z/°\u4e00-\u9fff]+)",
            text,
        )
        if not match:
            return None
        return self.convert(value=float(match.group(1)), from_unit=match.group(2), to_unit=match.group(3))

    def convert(self, *, value: float, from_unit: str, to_unit: str) -> UnitConversionResult:
        src = _normalize_unit(from_unit)
        dst = _normalize_unit(to_unit)
        result = _convert(value, src, dst)
        return UnitConversionResult(
            value=value,
            from_unit=src,
            to_unit=dst,
            result=result,
            result_text=_format_number(result),
        )


def format_unit_system_prompt(result: UnitConversionResult) -> str:
    return (
        "实时工具结果：单位换算\n"
        f"- 输入：{_format_number(result.value)} {result.from_unit}\n"
        f"- 输出：{result.result_text} {result.to_unit}\n"
        "回答单位换算题时必须基于以上工具结果，并说明换算关系。"
    )


def _normalize_unit(unit: str) -> str:
    raw = unit.strip().lower()
    normalized = UNIT_ALIASES.get(raw)
    if not normalized:
        raise ValueError(f"不支持的单位：{unit}")
    return normalized


def _convert(value: float, src: str, dst: str) -> float:
    if src in {"c", "f"} or dst in {"c", "f"}:
        return _convert_temperature(value, src, dst)
    src_kind, src_factor = LINEAR_FACTORS[src]
    dst_kind, dst_factor = LINEAR_FACTORS[dst]
    if src_kind != dst_kind:
        raise ValueError(f"不能从 {src} 换算到 {dst}")
    return value * src_factor / dst_factor


def _convert_temperature(value: float, src: str, dst: str) -> float:
    if src == dst:
        return value
    if src == "c" and dst == "f":
        return value * 9.0 / 5.0 + 32.0
    if src == "f" and dst == "c":
        return (value - 32.0) * 5.0 / 9.0
    raise ValueError(f"不能从 {src} 换算到 {dst}")


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-10:
        return str(int(round(value)))
    return f"{value:.10g}"
