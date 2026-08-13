"""Local time/date tool for deterministic current-time answers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


WEEKDAYS_ZH = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
TIME_MARKERS = ("今天几号", "今天星期", "现在几点", "当前时间", "今天日期", "现在时间", "几点了", "星期几")


@dataclass(frozen=True)
class TimeToolResult:
    now_iso: str
    date: str
    time: str
    weekday: str
    utc_offset: str
    source: str = "local-system-clock"

    def to_dict(self) -> dict[str, Any]:
        return {
            "now_iso": self.now_iso,
            "date": self.date,
            "time": self.time,
            "weekday": self.weekday,
            "utc_offset": self.utc_offset,
            "source": self.source,
        }


class LocalTimeTool:
    """Return local machine time with timezone offset."""

    def should_run(self, text: str) -> bool:
        return any(marker in text for marker in TIME_MARKERS)

    def lookup(self) -> TimeToolResult:
        now = datetime.now().astimezone()
        return TimeToolResult(
            now_iso=now.isoformat(timespec="seconds"),
            date=now.strftime("%Y-%m-%d"),
            time=now.strftime("%H:%M:%S"),
            weekday=WEEKDAYS_ZH[now.weekday()],
            utc_offset=now.strftime("%z")[:3] + ":" + now.strftime("%z")[3:],
        )


def format_time_system_prompt(result: TimeToolResult) -> str:
    return (
        "实时工具结果：本地时间\n"
        f"- 日期：{result.date}\n"
        f"- 时间：{result.time}\n"
        f"- 星期：{result.weekday}\n"
        f"- UTC 偏移：{result.utc_offset}\n"
        "回答日期或时间问题时必须基于以上工具结果。"
    )
