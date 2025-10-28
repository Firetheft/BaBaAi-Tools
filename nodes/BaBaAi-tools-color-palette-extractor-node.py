import torch
import webcolors
import logging
from numpy import ndarray
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

try:
    import colornamer
except ImportError:
    colornamer = None

class ColorPaletteExtractorNode:
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
        return {
            "required": {
                "input_image": ("IMAGE",),
            },
            "optional": {
                "num_colors": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 128,
                        "tooltip": "Number of colors to detect",
                    },
                ),
                "get_complementary": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_off": "Get Original Colors",
                        "label_on": "Get Complementary Colors",
                        "tooltip": "Get the complementary colors of the detected palette",
                    },
                ),
                "k_means_algorithm": (
                    ["lloyd", "elkan"],
                    {
                        "default": "lloyd",
                    },
                ),
                "accuracy": (
                    "INT",
                    {
                        "default": 60,
                        "display": "slider",
                        "min": 1,
                        "max": 100,
                        "tooltip": "Adjusts accuracy by changing number of iterations of the K-means algorithm",
                    },
                ),
                "exclude_colors": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Comma-separated list of colors to exclude from the output",
                    },
                ),
                "output_choices": (
                    ["plain_english_colors", "rgb_colors", "hex_colors", "xkcd_colors", "design_colors", "common_colors", "color_types", "color_families"],
                    {
                        "default": "plain_english_colors",
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
                )
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "output_text": (
                    "STRING",
                    {
                        "default": "",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("color_palette", "palette_image")
    OUTPUT_TOOLTIPS = ("This output returns the color information based on the user's selection.", "This output returns a generated color palette image.")
    FUNCTION = "main"
    OUTPUT_NODE = True
    CATEGORY = "📜BaBaAi Tools"

    def main(
        self,
        input_image: torch.Tensor,
        num_colors: int = 5,
        k_means_algorithm: str = "lloyd",
        accuracy: int = 80,
        get_complementary: bool = False,
        exclude_colors: str = "",
        output_text: str = "",
        output_choices: str = "plain_english_colors",
        unique_id=None,
        extra_pnginfo=None,
        palette_image_size: int = 128,
        palette_image_mode: str = "back_to_back"
    ) -> Tuple[str, torch.Tensor]:
        self.check_dependencies()

        if colornamer is None:
            self.logger.error("colornamer library not found.  XKCD, Design, Common, Type, and Family color outputs will be unavailable.")


        if exclude_colors.strip():
            self.exclude = exclude_colors.strip().split(",")
            self.exclude = [color.strip().lower() for color in self.exclude]
        else:
            self.exclude = []
        num_colors = max(1, num_colors)
        self.num_iterations = int(512 * (accuracy / 100))
        self.algorithm = k_means_algorithm if k_means_algorithm in ['lloyd', 'elkan'] else 'lloyd'
        self.webcolor_dict = {}
        for color_dict in [
            webcolors.CSS2_HEX_TO_NAMES,
            webcolors.CSS21_HEX_TO_NAMES,
            webcolors.CSS3_HEX_TO_NAMES,
            webcolors.HTML4_HEX_TO_NAMES,
        ]:
            self.webcolor_dict.update(color_dict)

        original_colors = self.interrogate_colors(
            input_image, num_colors, self.try_get_seed(extra_pnginfo)
        )
        rgb = self.ndarrays_to_rgb(original_colors)
        if get_complementary:
            rgb = self.rgb_to_complementary(rgb)

        plain_english_colors = [self.get_webcolor_name(color) for color in rgb]
        rgb_colors = [f"{color}" for color in rgb]
        hex_colors = [f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}" for color in rgb]

        colornamer_names = self.get_colornamer_names(rgb) if colornamer else [{"xkcd_color": "N/A", "design_color": "N/A", "common_color": "N/A", "color_type": "N/A", "color_family": "N/A"}] * len(rgb)

        xkcd_colors = [color["xkcd_color"] for color in colornamer_names]
        design_colors = [color["design_color"] for color in colornamer_names]
        common_colors = [color["common_color"] for color in colornamer_names]
        color_types = [color["color_type"] for color in colornamer_names]
        color_families = [color["color_family"] for color in colornamer_names]

        output_map = {
            "plain_english_colors": self.join_and_exclude(plain_english_colors),
            "rgb_colors": self.join_and_exclude(rgb_colors),
            "hex_colors": self.join_and_exclude(hex_colors),
            "xkcd_colors": self.join_and_exclude(xkcd_colors),
            "design_colors": self.join_and_exclude(design_colors),
            "common_colors": self.join_and_exclude(common_colors),
            "color_types": self.join_and_exclude(color_types),
            "color_families": self.join_and_exclude(color_families),
        }

        palette_image = self.generate_palette_image(rgb, palette_image_size, palette_image_mode)

        return (output_map[output_choices], palette_image)

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

    def ndarrays_to_rgb(self, colors: List[ndarray]) -> List[Tuple[int, int, int]]:
        return [(int(color[0]), int(color[1]), int(color[2])) for color in colors]

    def interrogate_colors(
        self, image: torch.Tensor, num_colors: int, seed: Optional[int] = None
    ) -> List[ndarray]:
        pixels = image.view(-1, image.shape[-1]).numpy()
        kmeans = KMeans(
            n_clusters=num_colors,
            algorithm=self.algorithm,
            max_iter=self.num_iterations,
            random_state=seed,
            n_init='auto'
        )
        colors = kmeans.fit(pixels).cluster_centers_ * 255
        return colors

    def get_webcolor_name(self, rgb: Tuple[int, int, int]) -> str:
        closest_match = None
        min_distance = float("inf")

        for hex, name in self.webcolor_dict.items():
            distance = sum(abs(a - b) for a, b in zip(rgb, webcolors.hex_to_rgb(hex)))
            if distance < min_distance:
                min_distance = distance
                closest_match = name

        return closest_match or "Unknown"

    def try_get_seed(self, extra_pnginfo: Dict[str, Any]) -> Optional[int]:
        try:
            for node in extra_pnginfo["workflow"]["nodes"]:
                if "Ksampler" not in node["type"]:
                    continue
                if isinstance(node["widgets_values"][0], (int, float)):
                    seed = node["widgets_values"][0]
                    if seed <= 0 or seed > 0xFFFFFFFF:
                        return None
                    return int(seed)
        except Exception:
            pass
        return None

    def generate_palette_image(self, colors: List[Tuple[int, int, int]], size: int, mode: str) -> torch.Tensor:
        num_colors = len(colors)
        if mode.lower() == "back_to_back":
            width = num_colors * size
            height = num_colors * size
            cell_height = height
        else:
            rows = int(num_colors**0.5)
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

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

NODE_CLASS_MAPPINGS = {
    "ColorPaletteExtractorNode": ColorPaletteExtractorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorPaletteExtractorNode": "调色板提取器"
}