import torch
import logging
import traceback
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class MultiAreaConditioning:

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "条件0": ("CONDITIONING", ),
            },
            "optional": {
                "条件1": ("CONDITIONING", ),
                "条件2": ("CONDITIONING", ),
                "条件3": ("CONDITIONING", )
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO", 
                "unique_id": "UNIQUE_ID"
            },
        }
    
    RETURN_TYPES = ("CONDITIONING", "INT", "INT")
    RETURN_NAMES = ("条件", "分辨率X", "分辨率Y")
    FUNCTION = "doStuff"
    CATEGORY = "📜BaBaAi Tools"

    DESCRIPTION = "Multi Area Conditioning with rotation support - fully compatible with ComfyUI v0.3.43"

    def _validate_area_params(self, area_params: List) -> Tuple[int, int, int, int, float, float]:

        try:
            if len(area_params) < 6:
                area_params = area_params + [1.0, 0.0] * (6 - len(area_params))
            
            x, y, w, h, strength, rotation = area_params[:6]
            
            x = max(0, int(x) if x is not None else 0)
            y = max(0, int(y) if y is not None else 0)
            w = max(8, int(w) if w is not None else 512)
            h = max(8, int(h) if h is not None else 512)
            strength = max(0.0, min(10.0, float(strength) if strength is not None else 1.0))
            rotation = max(-180.0, min(180.0, float(rotation) if rotation is not None else 0.0))
            
            return x, y, w, h, strength, rotation
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid area parameters, using defaults: {e}")
            return 0, 0, 512, 512, 1.0, 0.0

    def _extract_workflow_info(self, extra_pnginfo: Optional[Dict], unique_id: str) -> Tuple[List, int, int]:

        default_values = [
            [0, 0, 256, 192, 1.0, 0.0],
            [256, 0, 256, 192, 1.0, 0.0],
            [0, 192, 256, 192, 1.0, 0.0],
            [64, 128, 128, 256, 1.0, 0.0]
        ]
        default_resolution = (1024, 1024)
        
        try:
            if not extra_pnginfo or "workflow" not in extra_pnginfo:
                return default_values, *default_resolution
                
            workflow = extra_pnginfo["workflow"]
            if "nodes" not in workflow:
                return default_values, *default_resolution
                
            for node in workflow["nodes"]:
                if node.get("id") == int(unique_id):
                    properties = node.get("properties", {})
                    values = properties.get("values", default_values)
                    resolutionX = properties.get("width", 512)
                    resolutionY = properties.get("height", 512)

                    if not isinstance(values, list):
                        values = default_values

                    while len(values) < 4:
                        values.append([0, 0, 512, 512, 1.0, 0.0])
                    
                    return values, resolutionX, resolutionY
                    
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Failed to extract workflow info: {e}")
            
        return default_values, *default_resolution

    def _is_fullscreen_area(self, x: int, y: int, w: int, h: int, resX: int, resY: int) -> bool:

        return (x == 0 and y == 0 and w == resX and h == resY)

    def _apply_area_boundaries(self, x: int, y: int, w: int, h: int, resX: int, resY: int) -> Tuple[int, int, int, int]:

        if x + w > resX:
            w = max(8, resX - x)
        
        if y + h > resY:
            h = max(8, resY - y)

        w = ((w + 7) >> 3) << 3
        h = ((h + 7) >> 3) << 3
        
        return x, y, w, h

    def _process_conditioning_item(self, conditioning_item: Tuple, area_params: Tuple) -> Optional[List]:

        try:
            x, y, w, h, strength, rotation = area_params
            
            n = [conditioning_item[0], conditioning_item[1].copy()]

            n[1]['area'] = (h // 8, w // 8, y // 8, x // 8)
            n[1]['strength'] = strength
            n[1]['min_sigma'] = 0.0
            n[1]['max_sigma'] = 99.0

            n[1]['rotation'] = rotation
            
            return n
            
        except (IndexError, TypeError, AttributeError) as e:
            logger.warning(f"Failed to process conditioning item: {e}")
            return None

    def doStuff(self, extra_pnginfo: Optional[Dict], unique_id: str, **kwargs) -> Tuple[List, int, int]:

        try:
            values, resolutionX, resolutionY = self._extract_workflow_info(extra_pnginfo, unique_id)
            
            conditioning_results = []

            for k, (arg_name, conditioning) in enumerate(kwargs.items()):

                if k >= len(values):
                    break

                if conditioning is None:
                    continue

                if not self._validate_conditioning_data(conditioning):
                    continue

                area_params = self._validate_area_params(values[k])
                x, y, w, h, strength, rotation = area_params

                if self._is_fullscreen_area(x, y, w, h, resolutionX, resolutionY):

                    for item in conditioning:
                        conditioning_results.append(item)
                    continue

                x, y, w, h = self._apply_area_boundaries(x, y, w, h, resolutionX, resolutionY)

                if w <= 0 or h <= 0:
                    continue

                for item in conditioning:
                    processed_item = self._process_conditioning_item(item, (x, y, w, h, strength, rotation))
                    if processed_item:
                        conditioning_results.append(processed_item)
            
            return (conditioning_results, resolutionX, resolutionY)
            
        except Exception as e:
            logger.error(f"Error in doStuff: {e}")
            logger.error(traceback.format_exc())

            return ([], 512, 512)

    def _validate_conditioning_data(self, conditioning: Any) -> bool:

        try:
            if not conditioning or not isinstance(conditioning, (list, tuple)):
                return False
                
            if not torch.is_tensor(conditioning[0][0]):
                return False
                
            return True
            
        except (IndexError, TypeError, AttributeError):
            return False

NODE_CLASS_MAPPINGS = {
    "MultiAreaConditioning": MultiAreaConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiAreaConditioning": "可视化多区域条件",
}