#!/usr/bin/env python3
"""检查 funasr 模型 API"""
from funasr import AutoModel
import inspect

model = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    vad_model="fsmn-vad",
    device="cpu",
)

print("=== Model.generate 方法签名 ===")
print(inspect.signature(model.generate))
print("\n=== Model.generate 文档 ===")
print(model.generate.__doc__[:2000] if model.generate.__doc__ else "无文档")

print("\n=== Model 类型 ===")
print(type(model))
print("\n=== Model 属性 ===")
print([attr for attr in dir(model) if not attr.startswith('_')])