import comfy.model_management
import inspect
import logging
import torch
import torch.nn.functional as F
import nodes
from nodes import MAX_RESOLUTION

def vae_decode(vae, samples, use_tile, hook, tile_size=512, overlap=64):
    if use_tile:
        decoder = nodes.VAEDecodeTiled()
        if 'overlap' in inspect.signature(decoder.decode).parameters:
            pixels = decoder.decode(vae, samples, tile_size, overlap=overlap)[0]
        else:
            logging.warning("[Impact Pack] Your ComfyUI is outdated.")
            pixels = decoder.decode(vae, samples, tile_size)[0]
    else:
        pixels = nodes.VAEDecode().decode(vae, samples)[0]

    if hook is not None:
        pixels = hook.post_decode(pixels)

    return pixels

def vae_encode(vae, pixels, use_tile, hook, tile_size=512, overlap=64):
    if use_tile:
        encoder = nodes.VAEEncodeTiled()
        if 'overlap' in inspect.signature(encoder.encode).parameters:
            samples = encoder.encode(vae, pixels, tile_size, overlap=overlap)[0]
        else:
            logging.warning("[Impact Pack] Your ComfyUI is outdated.")
            samples = encoder.encode(vae, pixels, tile_size)[0]
    else:
        samples = nodes.VAEEncode().encode(vae, pixels)[0]

    if hook is not None:
        samples = hook.post_encode(samples)

    return samples

def latent_upscale_on_pixel_space_shape2(samples, scale_method, w, h, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    pixels = vae_decode(vae, samples, use_tile, hook, tile_size=tile_size, overlap=overlap)

    if save_temp_prefix is not None:
        nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(w), int(h), False)[0]

    old_pixels = pixels
    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return vae_encode(vae, pixels, use_tile, hook, tile_size=tile_size, overlap=overlap), old_pixels

def latent_upscale_on_pixel_space2(samples, scale_method, scale_factor, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    pixels = vae_decode(vae, samples, use_tile, hook, tile_size=tile_size, overlap=overlap)

    if save_temp_prefix is not None:
        nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

    w = pixels.shape[2] * scale_factor
    h = pixels.shape[1] * scale_factor
    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(w), int(h), False)[0]

    old_pixels = pixels
    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return vae_encode(vae, pixels, use_tile, hook, tile_size=tile_size, overlap=overlap), old_pixels

def latent_upscale_on_pixel_space_with_model_shape2(samples, scale_method, upscale_model, new_w, new_h, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    pixels = vae_decode(vae, samples, use_tile, hook, tile_size=tile_size, overlap=overlap)

    if save_temp_prefix is not None:
        nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

    w = pixels.shape[2]

    current_w = w
    while current_w < new_w:
        model_upscaler = nodes.NODE_CLASS_MAPPINGS['ImageUpscaleWithModel']()
        if hasattr(model_upscaler, 'execute'):
            pixels = model_upscaler.execute(upscale_model, pixels)[0]
        else:
            pixels = model_upscaler.upscale(upscale_model, pixels)[0]

        current_w = pixels.shape[2]
        if current_w == w:
            logging.info("[latent_upscale_on_pixel_space_with_model] x1 upscale model selected")
            break

    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(new_w), int(new_h), False)[0]

    old_pixels = pixels
    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return vae_encode(vae, pixels, use_tile, hook, tile_size=tile_size, overlap=overlap), old_pixels

def latent_upscale_on_pixel_space_with_model2(samples, scale_method, upscale_model, scale_factor, vae, use_tile=False,
    tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    pixels = vae_decode(vae, samples, use_tile, hook, tile_size=tile_size, overlap=overlap)

    if save_temp_prefix is not None:
        nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

    w = pixels.shape[2]
    h = pixels.shape[1]

    new_w = w * scale_factor
    new_h = h * scale_factor

    current_w = w
    while current_w < new_w:
        model_upscaler = nodes.NODE_CLASS_MAPPINGS['ImageUpscaleWithModel']()
        if hasattr(model_upscaler, 'execute'):
            pixels = model_upscaler.execute(upscale_model, pixels)[0]
        else:
            pixels = model_upscaler.upscale(upscale_model, pixels)[0]

        current_w = pixels.shape[2]
        if current_w == w:
            logging.info("[latent_upscale_on_pixel_space_with_model] x1 upscale model selected")
            break

    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(new_w), int(new_h), False)[0]

    old_pixels = pixels
    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return vae_encode(vae, pixels, use_tile, hook, tile_size=tile_size, overlap=overlap), old_pixels

class LatentPixelScale:
    upscale_methods = ["nearest-exact", "bilinear", "lanczos", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "samples": ("LATENT", ),
                     "scale_method": (s.upscale_methods,),
                     "scale_factor": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10000, "step": 0.05}),
                     "vae": ("VAE", ),
                     "use_tiled_vae": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                    }
                }

    RETURN_TYPES = ("LATENT", "IMAGE")
    FUNCTION = "doit"

    CATEGORY = "📜BaBaAi Tools"

    def doit(self, samples, scale_method, scale_factor, vae, use_tiled_vae, upscale_model_opt=None):
        if upscale_model_opt is None:
            latimg = latent_upscale_on_pixel_space2(samples, scale_method, scale_factor, vae, use_tile=use_tiled_vae)
        else:
            latimg = latent_upscale_on_pixel_space_with_model2(samples, scale_method, upscale_model_opt, scale_factor, vae, use_tile=use_tiled_vae)
        return latimg

NODE_CLASS_MAPPINGS = {
    "BaBaAiLatentPixelScale": LatentPixelScale
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BaBaAiLatentPixelScale": "缩放Latent（像素空间）"
}