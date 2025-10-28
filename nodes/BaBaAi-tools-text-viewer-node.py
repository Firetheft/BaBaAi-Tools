import torch
import json
from typing import Tuple, Any, Dict
import logging
import re
import html

logger = logging.getLogger(__name__)

class BaBaAiTextViewer:
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
            processed_text = re.sub(r'<[^>]+>', ' ', processed_text)
            processed_text = html.unescape(processed_text)
            processed_text = re.sub(r'^\s*[-*_]{3,}\s*$', '', processed_text, flags=re.MULTILINE)
            processed_text = re.sub(r'!\[.*?\]\(.*?\)', '', processed_text)
            processed_text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', processed_text)
            processed_text = re.sub(r'^\s*#+\s+(.*)\s*$', r'\1', processed_text, flags=re.MULTILINE)
            processed_text = re.sub(r'^\s*>\s+', '', processed_text, flags=re.MULTILINE)
            processed_text = re.sub(r'^\s*[-*+]\s+', '', processed_text, flags=re.MULTILINE)
            processed_text = re.sub(r'(\*\*\*|___)(.*?)\1', r'\2', processed_text)
            processed_text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', processed_text)
            processed_text = re.sub(r'(\*|_)(.*?)\1', r'\2', processed_text)
            processed_text = re.sub(r'(~~)(.*?)\1', r'\2', processed_text)
            processed_text = re.sub(r'`(.*?)`', r'\1', processed_text)
            processed_text = re.sub(r'\n{2,}', '\n', processed_text)
            processed_text = re.sub(r'(.+?)(?:[,，\s\n]+\1)+', r'\1', processed_text, flags=re.DOTALL)
            processed_text = re.sub(r'\s*,\s*', ', ', processed_text)
            processed_text = re.sub(r'(\s*,){2,}', ', ', processed_text)
            processed_text = re.sub(r'[ \t]{2,}', ' ', processed_text)
            processed_text = re.sub(r'^[ \t]+|[ \t]+$', '', processed_text, flags=re.MULTILINE)

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