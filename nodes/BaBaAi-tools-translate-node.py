import argostranslate.package
import argostranslate.translate
import argostranslate.settings
from typing import Dict, Any, Tuple
import os
from pathlib import Path

try:
    current_file_path = Path(__file__)
    comfyui_root_dir = current_file_path.parent.parent.parent.parent
    CUSTOM_PACKAGE_DIR = comfyui_root_dir / "models" / "argos-translate"

    os.makedirs(CUSTOM_PACKAGE_DIR, exist_ok=True)
    argostranslate.settings.package_data_dir = CUSTOM_PACKAGE_DIR

    if CUSTOM_PACKAGE_DIR not in argostranslate.settings.package_dirs:
        argostranslate.settings.package_dirs.insert(0, CUSTOM_PACKAGE_DIR)
        
    print(f"[TextTranslatorNode] Set custom model directory to: {CUSTOM_PACKAGE_DIR}")

except Exception as e:
    print(f"[TextTranslatorNode] Warning: Could not set custom model directory. Using default. Error: {e}")

class TextTranslatorNode:

    def __init__(self):
        self.language_mapping = {
            "en": "English",
            "zh": "Chinese",
            "ja": "Japanese", 
            "ko": "Korean",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "it": "Italian",
            "ru": "Russian",
            "ar": "Arabic",
            "pt": "Portuguese"
        }

        self.available_packages = None
        self.index_updated = False
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:

        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "source_language": (["auto", "en", "zh", "ja", "ko", "fr", "de", "es", "it", "ru", "ar", "pt"], {
                    "default": "auto"
                }),
                "target_language": (["en", "zh", "ja", "ko", "fr", "de", "es", "it", "ru", "ar", "pt"], {
                    "default": "en"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translated_text",)
    FUNCTION = "translate_text"
    CATEGORY = "📜BaBaAi Tools"
    
    def _check_and_install_package(self, from_code: str, to_code: str):

        if from_code == to_code:
            return

        try:
            installed_packages = argostranslate.package.get_installed_packages()
            package_exists = any(
                pkg.from_code == from_code and pkg.to_code == to_code
                for pkg in installed_packages
            )
            
            if package_exists:
                return

            print(f"[TextTranslatorNode] Package {from_code} -> {to_code} not found. Attempting to install...")

            if not self.index_updated:
                print("[TextTranslatorNode] Updating package index from remote...")
                argostranslate.package.update_package_index()
                self.index_updated = True

            if self.available_packages is None:
                self.available_packages = argostranslate.package.get_available_packages()

            package_to_install = next(
                (pkg for pkg in self.available_packages
                 if pkg.from_code == from_code and pkg.to_code == to_code),
                None
            )

            if package_to_install:
                print(f"[TextTranslatorNode] Installing language package: {from_code} -> {to_code}")
                argostranslate.package.install_from_path(package_to_install.download())
                print(f"[TextTranslatorNode] Installation complete: {from_code} -> {to_code}")
            else:
                print(f"[TextTranslatorNode] Warning: No available package found for {from_code} -> {to_code}")

        except Exception as e:
            print(f"[TextTranslatorNode] Warning: Could not check or install language package {from_code} -> {to_code}: {e}")

    
    def _detect_language(self, text: str) -> str:
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return "zh"
        elif any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text):
            return "ja"
        elif any('\uac00' <= char <= '\ud7af' for char in text):
            return "ko"
        else:
            return "en"
    
    def translate_text(self, text: str, source_language: str, target_language: str) -> Tuple[str]:
        try:
            if source_language == "auto":
                source_language = self._detect_language(text)
            
            if source_language == target_language:
                return (text,)
            
            self._check_and_install_package(source_language, target_language)
            
            translated_text = argostranslate.translate.translate(text, source_language, target_language)
            
            if not translated_text:
                if source_language != "en" and target_language != "en":
                    
                    self._check_and_install_package(source_language, "en")
                    english_text = argostranslate.translate.translate(text, source_language, "en")
                    
                    if english_text:
                        self._check_and_install_package("en", target_language)
                        translated_text = argostranslate.translate.translate(english_text, "en", target_language)
                
                if not translated_text:
                    return (f"Translation failed: No translation path available from {source_language} to {target_language}, even with 'en' fallback.",)
            
            return (translated_text,)
            
        except Exception as e:
            error_msg = f"Translation error: {str(e)}"
            print(f"[TextTranslatorNode] {error_msg}")
            if "No translation available" in error_msg:
                return (f"Translation failed: Could not find an installed model for {source_language} -> {target_language}. Check console for download errors.",)
            return (error_msg,)

NODE_CLASS_MAPPINGS = {
    "TextTranslatorNode": TextTranslatorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextTranslatorNode": "Argos文本翻译"
}