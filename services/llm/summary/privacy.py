"""Summary-specific exclusions; never retain credentials or diagnostic labels."""
from __future__ import annotations

import re

_SECRET = re.compile(
    r"(?:密码|口令|验证码|校验码|银行卡(?:号)?|身份证(?:号)?|password|passwd|otp|verification code|"
    r"bank account|credit card|api[_ -]?key)\s*(?:是|为|为你|[:：=]|is)?\s*[\w@!#$%&*.+/-]{3,}"
    r"|(?<!\d)\d{17}[\dXx](?!\d)|(?<!\d)\d{16,19}(?!\d)", re.I)
_LABEL = re.compile(r"智商|IQ\b|MBTI|人格(?:类型|障碍|评分)|(?:患有|诊断为|患上).{0,12}(?:症|障碍)|"
                    r"(?:用户|你|他|她).{0,8}(?:懒惰|不聪明|难相处)", re.I)


def contains_secret(text: str) -> bool:
    return bool(_SECRET.search(text))


def prohibited_conclusion(text: str) -> bool:
    return contains_secret(text) or bool(_LABEL.search(text))
