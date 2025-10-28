from functools import wraps
from typing import Callable
import latent_preview
import torch
from math import cos, sin, pi 

ORIGINAL_PREP: Callable = latent_preview.prepare_callback

RESHARPEN_STRENGTH: float = 0.0
RESHARPEN_SCALING_ALG: str = "Flat"
LATENT_CACHE: torch.Tensor = None


def apply_scaling(alg: str, decay: float, current_step: int, total_steps: int) -> float:
    if alg == "Flat":
        return decay

    if total_steps == 0:
        return decay
        
    ratio = float(current_step / total_steps)
    rad = ratio * pi / 2

    if alg == "Cos":
        mod = cos(rad)
    elif alg == "Sin":
        mod = sin(rad)
    elif alg == "1 - Cos":
        mod = 1 - cos(rad)
    elif alg == "1 - Sin":
        mod = 1 - sin(rad)
    else:
        mod = 1.0

    return decay * mod


def disable_resharpen():
    global RESHARPEN_STRENGTH, RESHARPEN_SCALING_ALG
    RESHARPEN_STRENGTH = 0.0
    RESHARPEN_SCALING_ALG = "Flat"

def hijack(PREP) -> Callable:

    @wraps(PREP)
    def prep_callback(*args, **kwargs):
        global LATENT_CACHE
        LATENT_CACHE = None

        original_callback: Callable = PREP(*args, **kwargs)

        if not RESHARPEN_STRENGTH:
            return original_callback

        print("Enable sharpening details！")

        @torch.inference_mode()
        @wraps(original_callback)
        def hijack_callback(step, x0, x, total_steps):

            if not RESHARPEN_STRENGTH:
                return original_callback(step, x0, x, total_steps)

            global LATENT_CACHE, RESHARPEN_SCALING_ALG, RESHARPEN_STRENGTH
            
            current_strength = apply_scaling(
                RESHARPEN_SCALING_ALG, 
                RESHARPEN_STRENGTH, 
                step, 
                total_steps
            )

            if current_strength != 0 and LATENT_CACHE is not None:
                try:
                    delta = x.detach().clone() - LATENT_CACHE
                    x += delta * current_strength
                except Exception as e:
                    print(f"[Resharpen Node] Error applying delta: {e}")

            LATENT_CACHE = x.detach().clone()
            return original_callback(step, x0, x, total_steps)

        return hijack_callback

    return prep_callback

latent_preview.prepare_callback = hijack(ORIGINAL_PREP)

class ResharpenDetailsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "details": (
                    "FLOAT",
                    {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1},
                ),
                "scaling_alg": (
                    ["Flat", "Cos", "Sin", "1 - Cos", "1 - Sin"], 
                    {"default": "Flat"}
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "hook"
    CATEGORY = "📜BaBaAi Tools"

    def hook(self, latent, details: float, scaling_alg: str):

        global RESHARPEN_STRENGTH, RESHARPEN_SCALING_ALG
        RESHARPEN_STRENGTH = details / -10.0
        RESHARPEN_SCALING_ALG = scaling_alg

        return (latent,)

NODE_CLASS_MAPPINGS = {
    "ResharpenDetailsNode": ResharpenDetailsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResharpenDetailsNode": "重新锐化细节"
}