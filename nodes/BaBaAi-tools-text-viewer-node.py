import torch
import json
from typing import Tuple, Any, Dict, Optional, List
import logging
import re
import html

logger = logging.getLogger(__name__)

def comprehensive_cleanup(text: str) -> str:
    processed_text = str(text)
    processed_text = processed_text.strip()
    processed_text = processed_text.replace('\\n', '\n')
    processed_text = re.sub(r'\r\n|\r', '\n', processed_text)
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

    all_commas = r',，'
    all_ends = r'。！？.'
    all_punct = all_commas + all_ends

    processed_text = re.sub(r'[ \t]*([%s])[ \t]*' % re.escape(all_punct), r'\1', processed_text)
    processed_text = re.sub(r'([%s])([%s \t])+' % (re.escape(all_commas), re.escape(all_commas)), r'\1', processed_text)
    processed_text = re.sub(r'([%s])([%s \t])+' % (re.escape(all_ends), re.escape(all_punct)), r'\1', processed_text)
    processed_text = re.sub(r'([%s])([%s])' % (re.escape(all_commas), re.escape(all_ends)), r'\2', processed_text)
    processed_text = re.sub(r'(.+?)(?:[,， \t]+\1)+', r'\1', processed_text, flags=re.IGNORECASE)
    processed_text = re.sub(r'[ \t]{2,}', ' ', processed_text)
    processed_text = re.sub(r'\n{3,}', '\n\n', processed_text)
    processed_text = re.sub(r'^[ \t]+|[ \t]+$', '', processed_text, flags=re.MULTILINE)
    processed_text = re.sub(r'^\s*[%s]+' % re.escape(all_punct), '', processed_text, flags=re.MULTILINE)

    return processed_text.strip()

class BaBaAiTextViewer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "text_input": ("STRING", {"forceInput": True, "default": ""}), 
                "auto_format": ("BOOLEAN", {
                    "default": True,
                    "label_on": "启用格式化",
                    "label_off": "保持原文本"
                }),
            },
            "optional": {
                "exclude_text": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "要排除的单词或短语，用英文逗号,分隔..."
                }),
                "find_text": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "要查找的单词或短语，用英文逗号,分隔..."
                }),
                "replace_text": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "要替换的单词或短语，用英文逗号,分隔..."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ('formatted_text',)
    OUTPUT_NODE = True
    FUNCTION = "process_and_display"
    CATEGORY = "📜BaBaAi Tools"

    def process_and_display(self, text_input: str, auto_format: bool, exclude_text: Optional[str] = "", find_text: Optional[str] = "", replace_text: Optional[str] = "") -> Dict:

        processed_text = str(text_input) 

        if auto_format:
            processed_text = comprehensive_cleanup(processed_text)

        if exclude_text and exclude_text.strip():
            exclusion_list = [term.strip() for term in exclude_text.split(',') if term.strip()]
            
            for term_to_exclude in exclusion_list:
                pattern = re.escape(term_to_exclude)
                processed_text = re.sub(pattern, '', processed_text, flags=re.IGNORECASE)
        
        if find_text and find_text.strip() and replace_text is not None:
            find_list = [term.strip() for term in find_text.split(',') if term.strip()]
            replace_list = [term.strip() for term in replace_text.split(',')]

            if len(find_list) == len(replace_list):
                for i in range(len(find_list)):
                    pattern = re.escape(find_list[i])
                    processed_text = re.sub(pattern, replace_list[i], processed_text, flags=re.IGNORECASE)
            else:
                logger.warning(f"[BaBaAiTextViewer] 查找和替换列表的长度不匹配。查找: {len(find_list)} 项, 替换: {len(replace_list)} 项。将跳过替换。")

        if auto_format:
            processed_text = comprehensive_cleanup(processed_text)

        formatted_text = processed_text.strip()

        return {
            "ui": {
                "text_output": [formatted_text]
            },
            "result": (formatted_text,)
        }

class BaBaAiConcatTextViewer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "auto_format": ("BOOLEAN", {
                    "default": True,
                    "label_on": "启用格式化",
                    "label_off": "保持原文本"
                }),
            },
            "optional": {
                "text_input_1": ("STRING", {"multiline": False, "default": "", "forceInput": False}),
                "text_input_2": ("STRING", {"multiline": False, "default": "", "forceInput": False}),
                "text_input_3": ("STRING", {"multiline": False, "default": "", "forceInput": False}),
                "text_input_4": ("STRING", {"multiline": False, "default": "", "forceInput": False}),
                "separator": ("STRING", {"multiline": False, "default": ""}),
                "exclude_text": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "要排除的单词或短语，用英文逗号,分隔..."
                }),
                "find_text": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "要查找的单词或短语，用英文逗号,分隔..."
                }),
                "replace_text": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "要替换的单词或短语，用英文逗号,分隔..."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ('formatted_text',)
    OUTPUT_NODE = True
    FUNCTION = "process_and_display"
    CATEGORY = "📜BaBaAi Tools"

    def process_and_display(self, auto_format: bool, 
                            text_input_1: Optional[str] = "",
                            text_input_2: Optional[str] = "",
                            text_input_3: Optional[str] = "",
                            text_input_4: Optional[str] = "",
                            separator: Optional[str] = "",
                            exclude_text: Optional[str] = "",
                            find_text: Optional[str] = "", 
                            replace_text: Optional[str] = "") -> Dict: 

        processed_separator = str(separator)
        if processed_separator:
            processed_separator = processed_separator.replace('\\n', '\n').replace('\\t', '\t')
            
        all_inputs = [text_input_1, text_input_2, text_input_3, text_input_4]
        non_empty_inputs = [str(i).strip() for i in all_inputs if i and str(i).strip()]
        
        if non_empty_inputs:
            processed_text = processed_separator.join(non_empty_inputs)
        else:
            processed_text = ""

        if not processed_text:
            return {
                "ui": {"text_output": [""]},
                "result": ("",)
            }

        if auto_format:
            processed_text = comprehensive_cleanup(processed_text)

        if exclude_text and exclude_text.strip():
            exclusion_list = [term.strip() for term in exclude_text.split(',') if term.strip()]
            
            for term_to_exclude in exclusion_list:
                pattern = re.escape(term_to_exclude)
                processed_text = re.sub(pattern, '', processed_text, flags=re.IGNORECASE)
        
        if find_text and find_text.strip() and replace_text is not None:
            find_list = [term.strip() for term in find_text.split(',') if term.strip()]
            replace_list = [term.strip() for term in replace_text.split(',')]

            if len(find_list) == len(replace_list):
                for i in range(len(find_list)):
                    pattern = re.escape(find_list[i])
                    processed_text = re.sub(pattern, replace_list[i], processed_text, flags=re.IGNORECASE)
            else:
                logger.warning(f"[BaBaAiConcatTextViewer] 查找和替换列表的长度不匹配。查找: {len(find_list)} 项, 替换: {len(replace_list)} 项。将跳过替换。")

        if auto_format:
            processed_text = comprehensive_cleanup(processed_text)

        formatted_text = processed_text.strip()

        return {
            "ui": {
                "text_output": [formatted_text]
            },
            "result": (formatted_text,)
        }

NODE_CLASS_MAPPINGS = {
    "BaBaAiTextViewer": BaBaAiTextViewer,
    "BaBaAiConcatTextViewer": BaBaAiConcatTextViewer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BaBaAiTextViewer": "文本查看器",
    "BaBaAiConcatTextViewer": "文本联结查看器"
}