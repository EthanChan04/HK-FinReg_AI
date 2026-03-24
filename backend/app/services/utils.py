"""
共享工具函数模块 (Utils)
从原 core_logic.py 中剥离出纯文本处理相关的工具代码。
"""
import re
from typing import Union
from datetime import datetime
from langchain_core.messages import AIMessage


def pii_scrubber(text: str) -> str:
    """隐私护盾层：脱敏 HKID 和电话号码"""
    if not text:
        return ""
    text = re.sub(r'[A-Z]\d{6}\([0-9A]\)', '[HKID REDACTED]', text)
    text = re.sub(r'\b[569]\d{7}\b', '[PHONE REDACTED]', text)
    return text


def extract_response_content(response) -> str:
    """从多种响应类型中提取纯文本"""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        if 'result' in response:
            return str(response['result']) if response['result'] else ""
        if 'content' in response:
            return str(response['content']) if response['content'] else ""
        return str(response)
    if isinstance(response, AIMessage):
        return response.content if response.content else ""
    if hasattr(response, 'content'):
        return str(response.content) if response.content else ""
    return str(response)


def format_output(text: Union[str, dict, AIMessage, None]) -> str:
    """清洗并规范化最终输出的 Markdown 文本"""
    text = extract_response_content(text)
    if not text:
        return ""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'\*\*\*+', '**', text)
    text = re.sub(r'---+', '---', text)
    text = re.sub(r'===+', '===', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^(\s*)[-*]\s*', r'\1- ', text, flags=re.MULTILINE)
    return text.strip()


def get_current_timestamp() -> str:
    """获取统一的 HKT 时间戳"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S HKT")
