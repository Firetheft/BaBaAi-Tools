import torch
import logging
import traceback
from typing import List, Tuple, Dict, Any, Optional

# 设置日志记录器
logger = logging.getLogger(__name__)

class MultiAreaConditioning:
    """
    支持最多4个条件输入，支持旋转角度控制
    Supports up to 4 conditioning inputs with rotation angle control
    """
    
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        定义输入类型 - v0.3.43兼容格式
        Define input types compatible with v0.3.43
        """
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
    
    # v0.3.43新增属性
    DESCRIPTION = "Multi Area Conditioning with rotation support - fully compatible with ComfyUI v0.3.43"

    def _validate_area_params(self, area_params: List) -> Tuple[int, int, int, int, float, float]:
        """
        验证和标准化区域参数
        Validate and normalize area parameters
        """
        try:
            if len(area_params) < 6:
                # 补充缺失的参数：strength默认1.0，rotation默认0.0
                area_params = area_params + [1.0, 0.0] * (6 - len(area_params))
            
            x, y, w, h, strength, rotation = area_params[:6]
            
            # 确保参数为数值类型并在有效范围内
            x = max(0, int(x) if x is not None else 0)
            y = max(0, int(y) if y is not None else 0)
            w = max(8, int(w) if w is not None else 512)  # 最小宽度8像素（8像素对齐）
            h = max(8, int(h) if h is not None else 512)  # 最小高度8像素
            strength = max(0.0, min(10.0, float(strength) if strength is not None else 1.0))
            rotation = max(-180.0, min(180.0, float(rotation) if rotation is not None else 0.0))
            
            return x, y, w, h, strength, rotation
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid area parameters, using defaults: {e}")
            return 0, 0, 512, 512, 1.0, 0.0

    def _extract_workflow_info(self, extra_pnginfo: Optional[Dict], unique_id: str) -> Tuple[List, int, int]:
        """
        提取工作流信息 - v0.3.43兼容
        Extract workflow information compatible with v0.3.43
        """
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
                    
                    # 验证和清理数据
                    if not isinstance(values, list):
                        values = default_values
                    
                    # 确保有4个区域配置
                    while len(values) < 4:
                        values.append([0, 0, 512, 512, 1.0, 0.0])
                    
                    return values, resolutionX, resolutionY
                    
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Failed to extract workflow info: {e}")
            
        return default_values, *default_resolution

    def _is_fullscreen_area(self, x: int, y: int, w: int, h: int, resX: int, resY: int) -> bool:
        """
        检查是否为全屏区域
        Check if area is fullscreen
        """
        return (x == 0 and y == 0 and w == resX and h == resY)

    def _apply_area_boundaries(self, x: int, y: int, w: int, h: int, resX: int, resY: int) -> Tuple[int, int, int, int]:
        """
        应用区域边界修正 - 确保8像素对齐
        Apply area boundary correction with 8-pixel alignment
        """
        # 边界修正
        if x + w > resX:
            w = max(8, resX - x)
        
        if y + h > resY:
            h = max(8, resY - y)
        
        # 8像素对齐
        w = ((w + 7) >> 3) << 3
        h = ((h + 7) >> 3) << 3
        
        return x, y, w, h

    def _process_conditioning_item(self, conditioning_item: Tuple, area_params: Tuple) -> Optional[List]:
        """
        处理单个conditioning项目
        Process single conditioning item
        """
        try:
            x, y, w, h, strength, rotation = area_params
            
            n = [conditioning_item[0], conditioning_item[1].copy()]
            
            # 应用区域参数（ComfyUI使用8像素单位）
            n[1]['area'] = (h // 8, w // 8, y // 8, x // 8)
            n[1]['strength'] = strength
            n[1]['min_sigma'] = 0.0
            n[1]['max_sigma'] = 99.0
            
            # 添加旋转角度信息（自定义属性，用于前端可视化）
            n[1]['rotation'] = rotation
            
            return n
            
        except (IndexError, TypeError, AttributeError) as e:
            logger.warning(f"Failed to process conditioning item: {e}")
            return None

    def doStuff(self, extra_pnginfo: Optional[Dict], unique_id: str, **kwargs) -> Tuple[List, int, int]:
        """
        主处理函数 - ComfyUI v0.3.43兼容
        Main processing function compatible with ComfyUI v0.3.43
        """
        try:
            # 提取工作流信息
            values, resolutionX, resolutionY = self._extract_workflow_info(extra_pnginfo, unique_id)
            
            conditioning_results = []
            
            # 处理所有conditioning输入
            for k, (arg_name, conditioning) in enumerate(kwargs.items()):
                # 边界检查
                if k >= len(values):
                    break
                
                # 如果conditioning为None（可选输入未连接），跳过
                if conditioning is None:
                    continue
                
                # 验证conditioning数据
                if not self._validate_conditioning_data(conditioning):
                    continue
                
                # 获取并验证区域参数
                area_params = self._validate_area_params(values[k])
                x, y, w, h, strength, rotation = area_params
                
                # 检查是否为全屏区域
                if self._is_fullscreen_area(x, y, w, h, resolutionX, resolutionY):
                    # 全屏区域直接添加
                    for item in conditioning:
                        conditioning_results.append(item)
                    continue
                
                # 应用边界修正
                x, y, w, h = self._apply_area_boundaries(x, y, w, h, resolutionX, resolutionY)
                
                # 检查修正后的区域是否有效
                if w <= 0 or h <= 0:
                    continue
                
                # 处理每个conditioning项目
                for item in conditioning:
                    processed_item = self._process_conditioning_item(item, (x, y, w, h, strength, rotation))
                    if processed_item:
                        conditioning_results.append(processed_item)
            
            return (conditioning_results, resolutionX, resolutionY)
            
        except Exception as e:
            logger.error(f"Error in doStuff: {e}")
            logger.error(traceback.format_exc())
            # 返回空结果以避免崩溃
            return ([], 512, 512)

    def _validate_conditioning_data(self, conditioning: Any) -> bool:
        """
        验证conditioning数据
        Validate conditioning data
        """
        try:
            if not conditioning or not isinstance(conditioning, (list, tuple)):
                return False
                
            # 检查第一个元素是否包含张量
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