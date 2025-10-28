import torch
import latent_preview
import comfy.samplers
import comfy.sample
import comfy.utils 
from math import cos, sin, pi 

def apply_scaling(alg: str, decay: float, current_step: int, total_steps: int) -> float:
    if alg == "Flat":
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

class ResharpenDetailsSamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "latent_image": ("LATENT", ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                "details": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "scaling_alg": (["Flat", "Cos", "Sin", "1 - Cos", "1 - Sin"], {"default": "Flat"}), 
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "📜BaBaAi Tools"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, 
               details, scaling_alg):
        
        is_video_model = False
        try:
            if hasattr(model.model, 'model_type') and 'video' in str(model.model.model_type).lower():
                 is_video_model = True
            model_class_name = model.__class__.__name__.lower()
            inner_model_class_name = getattr(model.model, '__class__', None).__name__.lower() if hasattr(model, 'model') else ''
            if 'video' in model_class_name or 'qwen' in model_class_name or 'video' in inner_model_class_name or 'qwen' in inner_model_class_name:
                is_video_model = True
        except Exception as e:
            print(f"[Resharpen Node] Warning: Could not determine model type. {e}")

        latent_samples = latent_image["samples"]

        if is_video_model and len(latent_samples.shape) == 4:
            print("[Resharpen Node] Detected 4D latent with video model. Reshaping to 5D [B, C, 1, H, W].")
            latent_samples = latent_samples.unsqueeze(2)

        base_strength = details / -10.0
        previewer = latent_preview.prepare_callback(model, steps)
        latent_cache = [None] 

        @torch.inference_mode()
        def resharpen_preview_callback(step, x0, x, total_steps):
            current_strength = apply_scaling(scaling_alg, base_strength, step, total_steps)
            
            if current_strength != 0:
                if latent_cache[0] is not None:
                    try:
                        delta = x.detach().clone() - latent_cache[0]
                        x += delta * current_strength 
                    except Exception as e:
                        pass 
                latent_cache[0] = x.detach().clone()
            
            return previewer(step, x0, x, total_steps)

        batch_inds = latent_image.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = comfy.sample.sample(
            model=model,
            noise=noise,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_samples, 
            denoise=denoise,
            disable_noise=False,
            start_step=None,
            last_step=None,
            force_full_denoise=False,
            noise_mask=None,
            callback=resharpen_preview_callback,
            disable_pbar=disable_pbar,
            seed=seed
        )

        return ({"samples": samples},) 

class ResharpenDetailsAdvancedSamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": (("enable", "disable"), ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "latent_image": ("LATENT", ),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (("disable", "enable"), ),
                
                "details": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "scaling_alg": (["Flat", "Cos", "Sin", "1 - Cos", "1 - Sin"], {"default": "Flat"}), 
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "📜BaBaAi Tools"

    def sample(self, model, add_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, 
               details, scaling_alg):
        
        is_video_model = False
        try:
            if hasattr(model.model, 'model_type') and 'video' in str(model.model.model_type).lower():
                 is_video_model = True
            model_class_name = model.__class__.__name__.lower()
            inner_model_class_name = getattr(model.model, '__class__', None).__name__.lower() if hasattr(model, 'model') else ''
            if 'video' in model_class_name or 'qwen' in model_class_name or 'video' in inner_model_class_name or 'qwen' in inner_model_class_name:
                is_video_model = True
        except Exception as e:
            print(f"[Resharpen Node] Warning: Could not determine model type. {e}")

        latent_samples = latent_image["samples"]

        if is_video_model and len(latent_samples.shape) == 4:
            print("[Resharpen Node] Detected 4D latent with video model. Reshaping to 5D [B, C, 1, H, W].")
            latent_samples = latent_samples.unsqueeze(2)

        base_strength = details / -10.0
        previewer = latent_preview.prepare_callback(model, steps)
        latent_cache = [None] 

        @torch.inference_mode()
        def resharpen_preview_callback(step, x0, x, total_steps):
            current_strength = apply_scaling(scaling_alg, base_strength, step, total_steps)
            
            if current_strength != 0:
                if latent_cache[0] is not None:
                    try:
                        delta = x.detach().clone() - latent_cache[0]
                        x += delta * current_strength 
                    except Exception as e:
                        pass 
                latent_cache[0] = x.detach().clone()
            
            return previewer(step, x0, x, total_steps)

        disable_noise = add_noise == "disable"
        force_full_denoise = return_with_leftover_noise == "disable"
        
        batch_inds = latent_image.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)
        
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = comfy.sample.sample(
            model=model,
            noise=noise,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_samples, 
            denoise=1.0, 
            disable_noise=disable_noise,
            start_step=start_at_step,
            last_step=end_at_step,
            force_full_denoise=force_full_denoise,
            noise_mask=None,
            callback=resharpen_preview_callback,
            disable_pbar=disable_pbar,
            seed=seed
        )

        return ({"samples": samples},)

NODE_CLASS_MAPPINGS = {
    "ResharpenDetailsSamplerNode": ResharpenDetailsSamplerNode,
    "ResharpenDetailsAdvancedSamplerNode": ResharpenDetailsAdvancedSamplerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResharpenDetailsSamplerNode": "重新锐化细节 (采样器)",
    "ResharpenDetailsAdvancedSamplerNode": "重新锐化细节 (高级采样器)"
}