import os
import requests
import json
from googletrans import Translator, LANGUAGES

### =====  GoogleTranslate Nodes [googletrans module]  ===== ###
translator = Translator()

google_translation_key = os.environ.get("GOOGLE_TRANSLATION_API_KEY")

# Translate used Google API_KEY
class TranslationResult:
    def __init__(self, text=""):
        self.text = text

    @staticmethod
    def translate_by_key(text, src, dest):
        url = f"https://translation.googleapis.com/language/translate/v2?key={google_translation_key}"
        data = {"q": text, "target": dest}

        try:
            resp = requests.post(url, data=data)
            resp.raise_for_status()  # Will raise an exception for HTTP error codes
            resp_data = resp.json()

            if "data" in resp_data and "translations" in resp_data["data"]:
                translations = resp_data["data"]["translations"]
                if translations and "translatedText" in translations[0]:
                    return TranslationResult(translations[0]["translatedText"])

        except requests.exceptions.RequestException as e:
            print(f"Error calling Google Translate API: {e}")
        except json.JSONDecodeError:
            print("Error decoding Google Translate API response.")
            
        return TranslationResult("")


def translate(prompt, srcTrans="auto", toTrans="en"):
    if not prompt or not prompt.strip():
        return ""

    try:
        if not google_translation_key:
            translation = translator.translate(prompt, src=srcTrans, dest=toTrans)
            return translation.text if hasattr(translation, "text") else ""
        else:
            translation = TranslationResult.translate_by_key(prompt, src=srcTrans, dest=toTrans)
            return translation.text
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        return prompt # Return original prompt on error

class GoogleTranslateCLIPNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "from_translate": (["auto"] + list(LANGUAGES.keys()), {"default": "auto"}),
                "to_translate": (list(LANGUAGES.keys()), {"default": "en"}),
                "text": ("STRING", {"multiline": True, "placeholder": "Input prompt"}),
                "clip": ("CLIP",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    FUNCTION = "execute_translation"
    CATEGORY = "📜BaBaAi Tools"

    def execute_translation(self, from_translate, to_translate, text, clip):
        text_translated = translate(text, from_translate, to_translate)
        tokens = clip.tokenize(text_translated)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], text_translated)


class GoogleTranslateTEXTNode(GoogleTranslateCLIPNode):

    @classmethod
    def INPUT_TYPES(cls):
        # Inherit and remove the 'clip' input
        types = super().INPUT_TYPES()
        del types["required"]["clip"]
        return types

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute_translation"
    CATEGORY = "📜BaBaAi Tools"

    def execute_translation(self, from_translate, to_translate, text):
        text_translated = translate(text, from_translate, to_translate)
        return (text_translated,)

# Node Mappings
NODE_CLASS_MAPPINGS = {
    "GoogleTranslateCLIPNode": GoogleTranslateCLIPNode,
    "GoogleTranslateTEXTNode": GoogleTranslateTEXTNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GoogleTranslateCLIPNode": "谷歌翻译-CLIP",
    "GoogleTranslateTEXTNode": "谷歌翻译-文本", 
}
### =====  GoogleTranslate Nodes [googletrans module] -> end ===== ###