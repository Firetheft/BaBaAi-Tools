import torch
import json
from typing import Tuple, Any, Dict
import logging
import re

logger = logging.getLogger(__name__)

class BaBaAiTextViewer:
    """
    一个用于BaBaAi Tools插件的文本查看器节点。
    它接收字符串输入，并提供一个开关来控制是否自动清理文本格式。
    处理后的结果会直接显示在节点界面上，同时提供一个字符串输出端口。
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "text_input": ("STRING", {"forceInput": True}),
                "auto_format": ("BOOLEAN", {
                    "default": True,
                    "label_on": "启用格式化",
                    "label_off": "保持原文本"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ('formatted_text',)
    OUTPUT_NODE = True
    FUNCTION = "process_and_display"
    CATEGORY = "📜BaBaAi Tools"

    def process_and_display(self, text_input: str, auto_format: bool) -> Dict:
        processed_text = text_input

        if auto_format:
            processed_text = processed_text.strip()
            processed_text = processed_text.replace('\\n\\n', '\n')
            processed_text = processed_text.replace('\\n', '\n')
            processed_text = re.sub(r'\n{2,}', '\n', processed_text)
            processed_text = re.sub(r'\s*,\s*', ', ', processed_text)
            processed_text = re.sub(r'(\s*,){2,}', ', ', processed_text)
            processed_text = re.sub(r'\s{2,}', ' ', processed_text)
        
        formatted_text = processed_text.strip()

        return {
            "ui": {
                "text_output": [formatted_text]
            },
            "result": (formatted_text,)
        }

NODE_CLASS_MAPPINGS = {
    "BaBaAiTextViewer": BaBaAiTextViewer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BaBaAiTextViewer": "文本查看器"
}