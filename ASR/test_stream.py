import base64
import json
import requests

# 准备测试音频
audio_path = r"C:\Users\32803\Desktop\录音\录音机-16点35分.wav"

with open(audio_path, 'rb') as f:
    audio_data = f.read()

audio_base64 = base64.b64encode(audio_data).decode()

# 发送请求
url = "http://localhost:8001/api/v1/asr/stream"
headers = {"Content-Type": "application/json"}
data = {
    "audio_base64": audio_base64,
    "filename": "test.wav",
    "lang": "zh",
    "use_itn": True,
    "is_final": False
}

print("正在发送请求...")
response = requests.post(url, headers=headers, json=data)
print(f"状态码: {response.status_code}")
print(f"响应头: {dict(response.headers)}")
print(f"响应内容:\n{response.text}")

# 解析结果
if response.status_code == 200 and response.text.startswith("data: "):
    result_str = response.text[6:]  # 去掉 "data: " 前缀
    try:
        result = json.loads(result_str)
        print("\n解析后的结果:")
        for item in result.get("result", []):
            print(f"  文本: {item.get('text')}")
            print(f"  语言: {item.get('language')}")
            print(f"  情绪: {item.get('emotion')} {item.get('emotion_emoji')}")
            if item.get('events'):
                print(f"  事件: {[e['label'] for e in item['events']]}")
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败: {e}")