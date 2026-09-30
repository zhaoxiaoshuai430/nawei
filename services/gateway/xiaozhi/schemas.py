from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BindRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str = Field(pattern=r"^\d{6}$")
    name: str = Field(default="Nawei Robot", min_length=1, max_length=80)


class DeviceUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, min_length=1, max_length=80)
    voice: str | None = Field(default=None, min_length=1, max_length=80, pattern=r"^[\w.-]+$")
    role: Literal["companion", "tutor"] | None = None
    enabled: bool | None = None


class AudioParams(BaseModel):
    format: Literal["opus"]
    sample_rate: Literal[16000]
    channels: Literal[1]
    frame_duration: Literal[60]


class Hello(BaseModel):
    type: Literal["hello"]
    version: Literal[1]
    transport: Literal["websocket"]
    audio_params: AudioParams
    output_audio_params: dict[str, int | str] | None = None

    def output_rate(self) -> int:
        if self.output_audio_params is None:
            return 24000
        if self.output_audio_params not in (
            {"format": "opus", "sample_rate": 16000, "channels": 1, "frame_duration": 60},
            {"format": "opus", "sample_rate": 24000, "channels": 1, "frame_duration": 60},
        ):
            raise ValueError("Unsupported output audio parameters")
        return int(self.output_audio_params["sample_rate"])
