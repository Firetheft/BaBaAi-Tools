import importlib


node_list = [
    "BaBaAi-tools-flux-prompt-enhance-node",
    "BaBaAi-tools-prompt-generator-node",
    "BaBaAi-tools-gemini-flash-node",
    "BaBaAi-tools-glm-flash-node",
    "BaBaAi-tools-color-palette-extractor-node",
    "BaBaAi-tools-color-palette-picker-node",
    "BaBaAi-tools-color-palette-transfer-node",
    "BaBaAi-tools-resharpen-details-node",
    "BaBaAi-tools-Multi-Area-Conditioning",
    "BaBaAi-tools-google-translate-node",
    "BaBaAi-tools-qwen-node",
    "BaBaAi-tools-audiocorp-node",
    "BaBaAi-tools-text-viewer-node"
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_name in node_list:
    imported_module = importlib.import_module(f".nodes.{module_name}", __name__)

    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **imported_module.NODE_DISPLAY_NAME_MAPPINGS}


WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]