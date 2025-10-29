import torch
import webcolors
import logging
from PIL import Image, ImageDraw
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import random

try:
    import colornamer
except ImportError:
    colornamer = None

from webcolors._definitions import (
    _CSS2_HEX_TO_NAMES,
    _CSS21_HEX_TO_NAMES,
    _CSS3_HEX_TO_NAMES,
    _HTML4_HEX_TO_NAMES,
)

try:
    from .meodai_colors import MEODAI_COLORS
    MEODAI_AVAILABLE = True
except ImportError:
    MEODAI_AVAILABLE = False
    logging.warning("[BabaColorPicker] Meodai color loader not found. 'meodai_color_names' will be unavailable.")


def hex_to_dec(inhex):
    try:
        rval = inhex[1:3]
        gval = inhex[3:5]
        bval = inhex[5:]
        rgbval = (int(rval, 16), int(gval, 16), int(bval, 16))
        return rgbval
    except (IndexError, ValueError):
        return (0, 0, 0)

class BabaColorInputNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color": ("BABA_COLOR", {"default": "#FFFFFF"}),
                "width": ("INT", {"default": 128, "min": 32, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 128, "min": 32, "max": 4096, "step": 8}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("hex_color", "color_image")
    FUNCTION = "get_color"
    CATEGORY = "📜BaBaAi Tools"
    OUTPUT_NODE = True

    def get_color(self, color, width, height):
        hex_string = color
        rgb_color = hex_to_dec(hex_string)
        image = Image.new('RGB', (width, height), color=rgb_color)
        tensor_image = pil2tensor(image)
        
        return (hex_string, tensor_image)

class ColorPalettePickerNode:
    dependencies_checked = False

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    @classmethod
    def check_dependencies(cls):
        if not cls.dependencies_checked:
            try:
                from ..install import check_and_install_dependencies
                if check_and_install_dependencies():
                    cls.logger.info("Dependencies were installed or updated. Please restart ComfyUI for changes to take effect.")
            except ImportError:
                cls.logger.warning("Could not import check_and_install_dependencies. Skipping dependency check.")
            cls.dependencies_checked = True

    @classmethod
    def INPUT_TYPES(s):
        
        output_options = [
            "plain_english_colors", 
            "rgb_colors", 
            "hex_colors", 
            "xkcd_colors", 
            "design_colors", 
            "common_colors", 
            "color_types", 
            "color_families"
        ]
        if MEODAI_AVAILABLE:
            output_options.append("meodai_color_names")
        
        return {
            "required": {
                "color1": ("BABA_COLOR", {"default": "#ffffff"}),
                "color2": ("BABA_COLOR", {"default": "#ffffff"}),
                "color3": ("BABA_COLOR", {"default": "#ffffff"}),
                "color4": ("BABA_COLOR", {"default": "#ffffff"}),
                "color5": ("BABA_COLOR", {"default": "#ffffff"}),
                "get_complementary": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_off": "Get Original Colors",
                        "label_on": "Get Complementary Colors",
                        "tooltip": "Get the complementary colors of the selected palette",
                    },
                ),
                "output_choices": (
                    output_options,
                    {
                        "default": "xkcd_colors",
                        "tooltip": "Select which color output to return",
                    },
                ),
                "palette_image_size": (
                    "INT",
                    {
                        "default": 128,
                        "min": 32,
                        "max": 512,
                        "tooltip": "Size of the generated palette image",
                    },
                ),
                "palette_image_mode": (
                    ["Chart", "back_to_back"],
                    {"default": "back_to_back"}
                ),
                "exclude_colors": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Comma-separated list of colors to exclude from the output",
                    },
                ),
                "randomize_colors": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_off": "Use Input Colors",
                        "label_on": "Generate Random Colors",
                        "tooltip": "Randomly generate 5 colors instead of using input colors",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1125899906842624,
                        "tooltip": "Random seed for generating colors. Leave blank or set to 0 for random seed.",
                    },
                ),
                "max_variation": (
                    "INT",
                    {
                        "default": 30,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "label": "Max Color Variation",
                        "tooltip": "Maximum amount of random variation applied to each color component (0-255)",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("color_palette", "palette_image")
    OUTPUT_TOOLTIPS = ("This output returns the color information based on the user's selection.", "This output returns a generated color palette image.")
    FUNCTION = "picker"
    OUTPUT_NODE = True
    CATEGORY = "📜BaBaAi Tools"

    def picker(
        self,
        color1, color2, color3, color4, color5, get_complementary, exclude_colors, output_choices, palette_image_size, palette_image_mode, randomize_colors, seed, max_variation
    ) -> Tuple[str, torch.Tensor]:
        self.check_dependencies()

        if colornamer is None:
            self.logger.error("colornamer library not found.  XKCD, Design, Common, Type, and Family color outputs will be unavailable.")

        self.webcolor_dict = {}

        for color_dict in [
            _CSS2_HEX_TO_NAMES,
            _CSS21_HEX_TO_NAMES,
            _CSS3_HEX_TO_NAMES,
            _HTML4_HEX_TO_NAMES,
        ]:
            self.webcolor_dict.update(color_dict)

        if exclude_colors.strip():
            self.exclude = exclude_colors.strip().split(",")
            self.exclude = [color.strip().lower() for color in self.exclude]
        else:
            self.exclude = []

        if randomize_colors:
            if seed != 0:  
                random.seed(seed)
            else:
                random.seed()

            rgb_colors = [hex_to_dec(c) for c in [color1, color2, color3, color4, color5]]
            rgb_colors = [self.add_randomness_to_color(color, max_variation) for color in rgb_colors]

        else:
            rgb_colors = [hex_to_dec(c) for c in [color1, color2, color3, color4, color5]]

        if get_complementary:
            rgb_colors = self.rgb_to_complementary(rgb_colors)

        plain_english_colors = [self.get_webcolor_name(color) for color in rgb_colors]
        hex_colors = [f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}" for color in rgb_colors]

        new_hex_colors_for_ui = hex_colors[:5]

        colornamer_names = self.get_colornamer_names(rgb_colors) if colornamer else [{"xkcd_color": "N/A", "design_color": "N/A", "common_color": "N/A", "color_type": "N/A", "color_family": "N/A"}] * len(rgb_colors)

        xkcd_colors = [color["xkcd_color"] for color in colornamer_names]
        design_colors = [color["design_color"] for color in colornamer_names]
        common_colors = [color["common_color"] for color in colornamer_names]
        color_types = [color["color_type"] for color in colornamer_names]
        color_families = [color["color_family"] for color in colornamer_names]

        meodai_color_names = ["N/A"] * len(rgb_colors)
        if MEODAI_AVAILABLE:
            meodai_color_names = [MEODAI_COLORS.get_closest_color_name(color) for color in rgb_colors]

        output_map = {
            "plain_english_colors": self.join_and_exclude(plain_english_colors),
            "rgb_colors": self.join_and_exclude([str(c) for c in rgb_colors]),
            "hex_colors": self.join_and_exclude(hex_colors),
            "xkcd_colors": self.join_and_exclude(xkcd_colors),
            "design_colors": self.join_and_exclude(design_colors),
            "common_colors": self.join_and_exclude(common_colors),
            "color_types": self.join_and_exclude(color_types),
            "color_families": self.join_and_exclude(color_families),
            "meodai_color_names": self.join_and_exclude(meodai_color_names),
        }

        palette_image = self.generate_palette_image(rgb_colors, palette_image_size, palette_image_mode)

        return {
            "ui": {
                "new_colors": new_hex_colors_for_ui
            },
            "result": (output_map[output_choices], palette_image)
        }

    def join_and_exclude(self, colors: List[str]) -> str:
        return ", ".join(
            [str(color) for color in colors if color.lower() not in self.exclude]
        )

    def get_colornamer_names(self, colors: List[Tuple[int, int, int]]) -> List[Dict[str, str]]:
        return [colornamer.get_color_from_rgb(color) for color in colors]

    def rgb_to_complementary(
        self, colors: List[Tuple[int, int, int]]
    ) -> List[Tuple[int, int, int]]:
        return [(255 - color[0], 255 - color[1], 255 - color[2]) for color in colors]

    def get_webcolor_name(self, rgb: Tuple[int, int, int]) -> str:
        closest_match = None
        min_distance = float("inf")

        for hex, name in self.webcolor_dict.items():
            distance = sum(abs(a - b) for a, b in zip(rgb, webcolors.hex_to_rgb(hex)))
            if distance < min_distance:
                min_distance = distance
                closest_match = name

        return closest_match or "Unknown"

    def generate_palette_image(self, colors: List[Tuple[int, int, int]], size: int, mode: str) -> torch.Tensor:
        num_colors = len(colors)
        if mode.lower() == "back_to_back":
            width = size * num_colors
            height = size * num_colors
            cell_height = height
        else:
            rows = int(num_colors ** 0.5)
            cols = int(np.ceil(num_colors / rows))
            width = cols * size
            height = rows * size
            cell_height = size

        palette = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(palette)

        for i, color in enumerate(colors):
            x = (i % (width // size)) * size
            y = (i // (width // size)) * cell_height
            draw.rectangle([x, y, x + size, y + cell_height], fill=color + (255,))

        return pil2tensor(palette)

    def generate_random_color(self) -> Tuple[int, int, int]:
        return tuple(random.randint(0, 255) for _ in range(3))

    def add_randomness_to_color(self, color: Tuple[int, int, int], max_variation: int = 30) -> Tuple[int, int, int]:
        r, g, b = color
        r = max(0, min(255, r + random.randint(-max_variation, max_variation)))
        g = max(0, min(255, g + random.randint(-max_variation, max_variation)))
        b = max(0, min(255, b + random.randint(-max_variation, max_variation)))
        return (r, g, b)

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

NODE_CLASS_MAPPINGS = {
    "ColorPalettePickerNode": ColorPalettePickerNode,
    "BabaColorInputNode": BabaColorInputNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorPalettePickerNode": "调色板选择器",
    "BabaColorInputNode": "颜色选择器"
}