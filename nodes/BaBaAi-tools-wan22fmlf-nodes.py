import torch
import json
from typing import List, Tuple, Optional, Any
import node_helpers
import comfy
import comfy.utils
import comfy.clip_vision
from nodes import MAX_RESOLUTION
import nodes
import inspect
import logging

def _encode_vae_helper(vae: Any, pixels: torch.Tensor, use_tiled_vae: bool, tile_size: int, node_name: str = "Wan22FMLF_Tiled") -> torch.Tensor:
    if use_tiled_vae and hasattr(nodes, 'VAEEncodeTiled'):
        try:
            encoder = nodes.VAEEncodeTiled()
            if 'overlap' in inspect.signature(encoder.encode).parameters:
                logging.info(f"[{node_name}] 正在使用 VAEEncodeTiled (带 overlap)，分块大小: {tile_size}")
                encoded_latent_dict = encoder.encode(vae, pixels, tile_size, overlap=64)[0]
            else:
                logging.warning(f"[{node_name}] 您的ComfyUI版本较旧，Tiled VAE overlap功能不可用。")
                encoded_latent_dict = encoder.encode(vae, pixels, tile_size)[0]
            
            return encoded_latent_dict["samples"]

        except Exception as e:
            print(f"[{node_name}] VAE分块编码失败，回退到标准编码。错误: {e}")
            encoded_latent_dict = nodes.VAEEncode().encode(vae, pixels)[0]
            return encoded_latent_dict["samples"]
    else:
        if use_tiled_vae:
             logging.warning(f"[{node_name}] 'use_tiled_vae' 已启用，但 VAEEncodeTiled 节点未找到。回退到标准编码。")

        encoded_latent_dict = nodes.VAEEncode().encode(vae, pixels)[0]
        return encoded_latent_dict["samples"]

class WanFirstMiddleLastFrameToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": MAX_RESOLUTION, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "mode": (["NORMAL", "SINGLE_PERSON"], {"default": "NORMAL"}),
                "start_image": ("IMAGE",),
                "middle_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "middle_frame_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "high_noise_mid_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "low_noise_start_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "low_noise_mid_strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "low_noise_end_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "clip_vision_start_image": ("CLIP_VISION_OUTPUT",),
                "clip_vision_middle_image": ("CLIP_VISION_OUTPUT",),
                "clip_vision_end_image": ("CLIP_VISION_OUTPUT",),
                "use_tiled_vae": ("BOOLEAN", {"default": False, "label_on": "启用分块编码", "label_off": "禁用分块编码"}),
                "vae_tile_size": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
            },
        }
        return inputs

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive_high_noise", "positive_low_noise", "negative", "latent")
    FUNCTION = "generate"
    CATEGORY = "📜BaBaAi Tools/Wan22FMLF"

    def generate(
        self,
        positive: Tuple[Any, ...], negative: Tuple[Any, ...], vae: Any,
        width: int, height: int, length: int, batch_size: int,
        mode: str = "NORMAL",
        start_image: Optional[torch.Tensor] = None,
        middle_image: Optional[torch.Tensor] = None,
        end_image: Optional[torch.Tensor] = None,
        middle_frame_ratio: float = 0.5,
        high_noise_mid_strength: float = 0.8,
        low_noise_start_strength: float = 1.0,
        low_noise_mid_strength: float = 0.2,
        low_noise_end_strength: float = 1.0,
        clip_vision_start_image: Optional[Any] = None,
        clip_vision_middle_image: Optional[Any] = None,
        clip_vision_end_image: Optional[Any] = None,
        use_tiled_vae: bool = False,
        vae_tile_size: int = 512
    ) -> Tuple[Tuple[Any, ...], Tuple[Any, ...], Tuple[Any, ...], dict]:

        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((length - 1) // 4) + 1
        device = comfy.model_management.intermediate_device()

        latent = torch.zeros([batch_size, latent_channels, latent_t, height // spacial_scale, width // spacial_scale], device=device)

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if middle_image is not None:
            middle_image = comfy.utils.common_upscale(middle_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if end_image is not None:
            end_image = comfy.utils.common_upscale(end_image[-length:].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)

        image = torch.ones((length, height, width, 3), device=device) * 0.5
        mask_base = torch.ones((1, 1, latent_t * 4, latent.shape[-2], latent.shape[-1]), device=device)

        middle_idx = self._calculate_aligned_position(middle_frame_ratio, length)
        middle_idx = max(4, min(middle_idx, length - 5))

        mask_high_noise = mask_base.clone()
        mask_low_noise = mask_base.clone()

        if start_image is not None:
            image[:start_image.shape[0]] = start_image
            mask_high_noise[:, :, :start_image.shape[0] + 3] = 0.0
            mask_low_noise[:, :, :start_image.shape[0] + 3] = 1.0 - low_noise_start_strength
        if middle_image is not None:
            image[middle_idx:middle_idx + 1] = middle_image
            start_range = max(0, middle_idx)
            end_range = min(length, middle_idx + 4)
            mask_high_noise[:, :, start_range:end_range] = 1.0 - high_noise_mid_strength
            mask_low_noise[:, :, start_range:end_range] = 1.0 - low_noise_mid_strength
        if end_image is not None:
            image[-end_image.shape[0]:] = end_image
            mask_high_noise[:, :, -end_image.shape[0]:] = 0.0
            mask_low_noise[:, :, -end_image.shape[0]:] = 1.0 - low_noise_end_strength

        concat_latent_image = _encode_vae_helper(vae, image[:, :, :, :3], use_tiled_vae, vae_tile_size, "WanFirstMiddleLastFrame")

        if mode == "SINGLE_PERSON":
            image_low_only = torch.ones((length, height, width, 3), device=device) * 0.5
            if start_image is not None:
                image_low_only[:start_image.shape[0]] = start_image

            concat_latent_image_low = _encode_vae_helper(vae, image_low_only[:, :, :, :3], use_tiled_vae, vae_tile_size, "WanFirstMiddleLastFrame")
        elif low_noise_start_strength == 0.0 or low_noise_mid_strength == 0.0 or low_noise_end_strength == 0.0:
            image_low_only = torch.ones((length, height, width, 3), device=device) * 0.5
            if start_image is not None and low_noise_start_strength > 0.0:
                image_low_only[:start_image.shape[0]] = start_image
            if middle_image is not None and low_noise_mid_strength > 0.0:
                image_low_only[middle_idx:middle_idx + 1] = middle_image
            if end_image is not None and low_noise_end_strength > 0.0:
                image_low_only[-end_image.shape[0]:] = end_image

            concat_latent_image_low = _encode_vae_helper(vae, image_low_only[:, :, :, :3], use_tiled_vae, vae_tile_size, "WanFirstMiddleLastFrame")
        else:
            concat_latent_image_low = concat_latent_image

        mask_high_reshaped = mask_high_noise.view(1, mask_high_noise.shape[2] // 4, 4, mask_high_noise.shape[3], mask_high_noise.shape[4]).transpose(1, 2)
        mask_low_reshaped = mask_low_noise.view(1, mask_low_noise.shape[2] // 4, 4, mask_low_noise.shape[3], mask_low_noise.shape[4]).transpose(1, 2)

        positive_high_noise = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask_high_reshaped})
        positive_low_noise = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image_low, "concat_mask": mask_low_reshaped})

        clip_vision_output = self._merge_clip_vision_outputs(clip_vision_start_image, clip_vision_middle_image, clip_vision_end_image)
        if clip_vision_output is not None:
            positive_low_noise = node_helpers.conditioning_set_values(positive_low_noise, {"clip_vision_output": clip_vision_output})

        out_latent = {"samples": latent}
        return (positive_high_noise, positive_low_noise, negative, out_latent)

    def _calculate_aligned_position(self, ratio: float, total_frames: int) -> int:
        desired_idx = int(total_frames * ratio)
        latent_idx = desired_idx // 4
        aligned_idx = latent_idx * 4
        return max(0, min(aligned_idx, total_frames - 1))

    def _merge_clip_vision_outputs(self, *outputs: Any) -> Optional[Any]:
        valid_outputs = [o for o in outputs if o is not None]
        if not valid_outputs: return None
        if len(valid_outputs) == 1: return valid_outputs[0]
        all_states = [o.penultimate_hidden_states for o in valid_outputs]
        combined_states = torch.cat(all_states, dim=-2)
        result = comfy.clip_vision.Output()
        result.penultimate_hidden_states = combined_states
        return result

class WanMultiFrameRefToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": MAX_RESOLUTION, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "ref_images": ("IMAGE",),
            },
            "optional": {
                "mode": (["NORMAL", "SINGLE_PERSON"], {"default": "NORMAL"}),
                "ref_positions": ("STRING", {"default": "", "multiline": False}),
                "ref_strength_high": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "ref_strength_low": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "end_frame_strength_high": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "end_frame_strength_low": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "use_tiled_vae": ("BOOLEAN", {"default": False, "label_on": "启用分块编码", "label_off": "禁用分块编码"}),
                "vae_tile_size": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive_high_noise", "positive_low_noise", "negative", "latent")
    FUNCTION = "generate"
    CATEGORY = "📜BaBaAi Tools/Wan22FMLF"

    def generate(self, positive: Tuple[Any, ...], negative: Tuple[Any, ...],
                 vae: Any, width: int, height: int, length: int, batch_size: int,
                 ref_images: torch.Tensor, mode: str = "NORMAL",
                 ref_positions: str = "", ref_strength_high: float = 0.8,
                 ref_strength_low: float = 0.2, end_frame_strength_high: float = 1.0,
                 end_frame_strength_low: float = 1.0, clip_vision_output: Optional[Any] = None,
                 use_tiled_vae: bool = False, vae_tile_size: int = 512
                 ) -> Tuple[Tuple[Any, ...], Tuple[Any, ...], Tuple[Any, ...], dict]:
        
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((length - 1) // 4) + 1
        device = comfy.model_management.intermediate_device()
        
        latent = torch.zeros([batch_size, latent_channels, latent_t, height // spacial_scale, width // spacial_scale], device=device)
        
        imgs = self._resize_images(ref_images, width, height, device)
        n_imgs = imgs.shape[0]
        positions = self._parse_positions(ref_positions, n_imgs, length)
        
        def align_position(pos: int, total_frames: int) -> int:
            latent_idx = pos // 4
            aligned_pos = latent_idx * 4
            return max(0, min(aligned_pos, total_frames - 1))
        
        aligned_positions = [align_position(int(p), length) for p in positions]
        
        for i in range(1, len(aligned_positions)):
            if aligned_positions[i] <= aligned_positions[i-1] + 3:
                aligned_positions[i] = min(aligned_positions[i-1] + 4, length - 1)
        
        image = torch.ones((length, height, width, 3), device=device) * 0.5
        mask_base = torch.ones((1, 1, latent_t * 4, latent.shape[-2], latent.shape[-1]), device=device)
        mask_high_noise = mask_base.clone()
        mask_low_noise = mask_base.clone()
        
        for i, pos in enumerate(aligned_positions):
            frame_idx = int(pos)
            if i == 0:
                image[frame_idx:frame_idx + 1] = imgs[i]
                mask_high_noise[:, :, frame_idx:frame_idx + 4] = 0.0
                mask_low_noise[:, :, frame_idx:frame_idx + 4] = 0.0
            elif i == n_imgs - 1:
                image[-1:] = imgs[i]
                mask_high_noise[:, :, -4:] = 1.0 - end_frame_strength_high
                mask_low_noise[:, :, -4:] = 1.0 - end_frame_strength_low
            else:
                image[frame_idx:frame_idx + 1] = imgs[i]
                start_range = max(0, frame_idx)
                end_range = min(length, frame_idx + 4)
                mask_high_noise[:, :, start_range:end_range] = 1.0 - ref_strength_high
                mask_low_noise[:, :, start_range:end_range] = 1.0 - ref_strength_low

        if mode == "SINGLE_PERSON":
            concat_latent_image_high = _encode_vae_helper(vae, image[:, :, :, :3], use_tiled_vae, vae_tile_size, "WanMultiFrameRef")
        else:
            need_selective_image_high = (ref_strength_high == 0.0) or (end_frame_strength_high == 0.0)
            if need_selective_image_high:
                image_high_only = torch.ones((length, height, width, 3), device=device) * 0.5
                if n_imgs >= 1: image_high_only[int(aligned_positions[0]):int(aligned_positions[0]) + 1] = imgs[0]
                if ref_strength_high > 0.0:
                    for i in range(1, n_imgs - 1): image_high_only[int(aligned_positions[i]):int(aligned_positions[i]) + 1] = imgs[i]
                if n_imgs >= 2 and end_frame_strength_high > 0.0: image_high_only[-1:] = imgs[-1]
                concat_latent_image_high = _encode_vae_helper(vae, image_high_only[:, :, :, :3], use_tiled_vae, vae_tile_size, "WanMultiFrameRef")
            else:
                concat_latent_image_high = _encode_vae_helper(vae, image[:, :, :, :3], use_tiled_vae, vae_tile_size, "WanMultiFrameRef")
        
        if mode == "SINGLE_PERSON":
            mask_low_noise = mask_base.clone()
            if n_imgs >= 1: mask_low_noise[:, :, int(aligned_positions[0]):int(aligned_positions[0]) + 4] = 0.0
            if n_imgs >= 2: mask_low_noise[:, :, -4:] = 1.0 - end_frame_strength_low
            image_low_only = torch.ones((length, height, width, 3), device=device) * 0.5
            if n_imgs >= 1: image_low_only[int(aligned_positions[0]):int(aligned_positions[0]) + 1] = imgs[0]
            if n_imgs >= 2 and end_frame_strength_low > 0.0: image_low_only[-1:] = imgs[-1]
            concat_latent_image_low = _encode_vae_helper(vae, image_low_only[:, :, :, :3], use_tiled_vae, vae_tile_size, "WanMultiFrameRef")
        else:
            need_selective_image = (ref_strength_low == 0.0) or (end_frame_strength_low == 0.0)
            if need_selective_image:
                image_low_only = torch.ones((length, height, width, 3), device=device) * 0.5
                if n_imgs >= 1: image_low_only[int(aligned_positions[0]):int(aligned_positions[0]) + 1] = imgs[0]
                if ref_strength_low > 0.0:
                    for i in range(1, n_imgs - 1): image_low_only[int(aligned_positions[i]):int(aligned_positions[i]) + 1] = imgs[i]
                if n_imgs >= 2 and end_frame_strength_low > 0.0: image_low_only[-1:] = imgs[-1]
                concat_latent_image_low = _encode_vae_helper(vae, image_low_only[:, :, :, :3], use_tiled_vae, vae_tile_size, "WanMultiFrameRef")
            else:
                concat_latent_image_low = _encode_vae_helper(vae, image[:, :, :, :3], use_tiled_vae, vae_tile_size, "WanMultiFrameRef")
        
        mask_high_reshaped = mask_high_noise.view(1, mask_high_noise.shape[2] // 4, 4, mask_high_noise.shape[3], mask_high_noise.shape[4]).transpose(1, 2)
        mask_low_reshaped = mask_low_noise.view(1, mask_low_noise.shape[2] // 4, 4, mask_low_noise.shape[3], mask_low_noise.shape[4]).transpose(1, 2)
        
        positive_high_noise = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image_high, "concat_mask": mask_high_reshaped})
        positive_low_noise = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image_low, "concat_mask": mask_low_reshaped})
        
        if clip_vision_output is not None:
            positive_low_noise = node_helpers.conditioning_set_values(positive_low_noise, {"clip_vision_output": clip_vision_output})
        
        return (positive_high_noise, positive_low_noise, negative, {"samples": latent})

    def _resize_images(self, images: torch.Tensor, width: int, height: int, device: torch.device) -> torch.Tensor:
        images = images.to(device)
        x = images.movedim(-1, 1)
        x = comfy.utils.common_upscale(x, width, height, "bilinear", "center")
        x = x.movedim(1, -1)
        return x[..., :3] if x.shape[-1] == 4 else x

    def _parse_positions(self, pos_str: str, n_imgs: int, length: int) -> List[int]:
        positions = []
        s = (pos_str or "").strip()
        if s:
            try:
                if s.startswith("["): positions = json.loads(s)
                else: positions = [float(x.strip()) for x in s.split(",") if x.strip()]
            except Exception: positions = []
        
        if not positions:
            positions = [0] if n_imgs <= 1 else [i * (length - 1) / (n_imgs - 1) for i in range(n_imgs)]
        
        converted_positions = [int(p * (length - 1)) if 0 <= p < 2.0 else int(p) for p in positions]
        converted_positions = [max(0, min(length - 1, p)) for p in converted_positions]
        
        if len(converted_positions) > n_imgs: converted_positions = converted_positions[:n_imgs]
        elif len(converted_positions) < n_imgs: converted_positions.extend([converted_positions[-1]] * (n_imgs - len(converted_positions)))
        
        return converted_positions

class WanFourFrameReferenceUltimate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": MAX_RESOLUTION, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "mode": (["NORMAL", "SINGLE_PERSON"], {"default": "NORMAL"}),
                "frame_1_image": ("IMAGE",),
                "frame_2_image": ("IMAGE",),
                "frame_2_ratio": ("FLOAT", {"default": 0.33, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "frame_2_strength_high": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "frame_2_strength_low": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "enable_frame_2": (["disable", "enable"], {"default": "enable"}),
                "frame_3_image": ("IMAGE",),
                "frame_3_ratio": ("FLOAT", {"default": 0.67, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "frame_3_strength_high": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "frame_3_strength_low": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "enable_frame_3": (["disable", "enable"], {"default": "enable"}),
                "frame_4_image": ("IMAGE",),
                "clip_vision_frame_1": ("CLIP_VISION_OUTPUT",),
                "clip_vision_frame_2": ("CLIP_VISION_OUTPUT",),
                "clip_vision_frame_3": ("CLIP_VISION_OUTPUT",),
                "clip_vision_frame_4": ("CLIP_VISION_OUTPUT",),
                "use_tiled_vae": ("BOOLEAN", {"default": False, "label_on": "启用分块编码", "label_off": "禁用分块编码"}),
                "vae_tile_size": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive_high_noise", "positive_low_noise", "negative", "latent")
    FUNCTION = "generate"
    CATEGORY = "📜BaBaAi Tools/Wan22FMLF"

    def generate(self, positive: Tuple[Any, ...], negative: Tuple[Any, ...],
                 vae: Any, width: int, height: int, length: int, batch_size: int,
                 mode: str = "NORMAL",
                 frame_1_image: Optional[torch.Tensor] = None,
                 frame_2_image: Optional[torch.Tensor] = None,
                 frame_2_ratio: float = 0.33, frame_2_strength_high: float = 0.8,
                 frame_2_strength_low: float = 0.2, enable_frame_2: str = "enable",
                 frame_3_image: Optional[torch.Tensor] = None,
                 frame_3_ratio: float = 0.67, frame_3_strength_high: float = 0.8,
                 frame_3_strength_low: float = 0.2, enable_frame_3: str = "enable",
                 frame_4_image: Optional[torch.Tensor] = None,
                 clip_vision_frame_1: Optional[Any] = None,
                 clip_vision_frame_2: Optional[Any] = None,
                 clip_vision_frame_3: Optional[Any] = None,
                 clip_vision_frame_4: Optional[Any] = None,
                 use_tiled_vae: bool = False, vae_tile_size: int = 512
                 ) -> Tuple[Tuple[Any, ...], Tuple[Any, ...], Tuple[Any, ...], dict]:
        
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((length - 1) // 4) + 1
        device = comfy.model_management.intermediate_device()
        
        latent = torch.zeros([batch_size, latent_channels, latent_t, height // spacial_scale, width // spacial_scale], device=device)
        
        if frame_1_image is not None: frame_1_image = comfy.utils.common_upscale(frame_1_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if frame_2_image is not None: frame_2_image = comfy.utils.common_upscale(frame_2_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if frame_3_image is not None: frame_3_image = comfy.utils.common_upscale(frame_3_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if frame_4_image is not None: frame_4_image = comfy.utils.common_upscale(frame_4_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        
        image = torch.ones((length, height, width, 3), device=device) * 0.5
        mask_base = torch.ones((1, 1, latent_t * 4, latent.shape[-2], latent.shape[-1]), device=device)
        
        def calculate_aligned_position(ratio: float, total_frames: int) -> Tuple[int, int]:
            desired_pixel_idx = int(total_frames * ratio)
            latent_idx = desired_pixel_idx // 4
            aligned_pixel_idx = latent_idx * 4
            return max(0, min(aligned_pixel_idx, total_frames - 1)), latent_idx
        
        frame_1_idx, _ = 0, 0
        frame_2_idx, _ = calculate_aligned_position(frame_2_ratio, length)
        frame_3_idx, _ = calculate_aligned_position(frame_3_ratio, length)
        frame_4_idx, _ = calculate_aligned_position((length - 1) / length, length)
        
        if frame_2_idx <= frame_1_idx + 4: frame_2_idx = frame_1_idx + 4
        if frame_3_idx <= frame_2_idx + 4: frame_3_idx = frame_2_idx + 4
        if frame_4_idx <= frame_3_idx + 4: frame_4_idx = frame_3_idx + 4
        
        mask_high_noise = mask_base.clone()
        mask_low_noise = mask_base.clone()
        
        if frame_1_image is not None:
            image[:frame_1_image.shape[0]] = frame_1_image
            mask_high_noise[:, :, :frame_1_image.shape[0] + 3] = 0.0
            mask_low_noise[:, :, :frame_1_image.shape[0] + 3] = 0.0
        
        if frame_2_image is not None and enable_frame_2 == "enable":
            image[frame_2_idx:frame_2_idx + frame_2_image.shape[0]] = frame_2_image
            start_range = max(0, frame_2_idx)
            end_range = min(length, frame_2_idx + frame_2_image.shape[0] + 3)
            mask_high_noise[:, :, start_range:end_range] = 1.0 - frame_2_strength_high
            mask_low_noise[:, :, start_range:end_range] = 1.0 - frame_2_strength_low
        
        if frame_3_image is not None and enable_frame_3 == "enable":
            image[frame_3_idx:frame_3_idx + frame_3_image.shape[0]] = frame_3_image
            start_range = max(0, frame_3_idx)
            end_range = min(length, frame_3_idx + frame_3_image.shape[0] + 3)
            mask_high_noise[:, :, start_range:end_range] = 1.0 - frame_3_strength_high
            mask_low_noise[:, :, start_range:end_range] = 1.0 - frame_3_strength_low
        
        if frame_4_image is not None:
            image[frame_4_idx:frame_4_idx + frame_4_image.shape[0]] = frame_4_image
            mask_high_noise[:, :, frame_4_idx:frame_4_idx + frame_4_image.shape[0] + 3] = 0.0
            mask_low_noise[:, :, frame_4_idx:frame_4_idx + frame_4_image.shape[0] + 3] = 0.0
        
        concat_latent_image_high = _encode_vae_helper(vae, image[:, :, :, :3], use_tiled_vae, vae_tile_size, "WanFourFrameRef")
        
        if mode == "SINGLE_PERSON":
            mask_low_noise = mask_base.clone()
            if frame_1_image is not None:
                mask_low_noise[:, :, :frame_1_image.shape[0] + 3] = 0.0
            image_low_only = torch.ones((length, height, width, 3), device=device) * 0.5
            if frame_1_image is not None:
                image_low_only[:frame_1_image.shape[0]] = frame_1_image

            concat_latent_image_low = _encode_vae_helper(vae, image_low_only[:, :, :, :3], use_tiled_vae, vae_tile_size, "WanFourFrameRef")
        else:
            frame_2_strength = frame_2_strength_low if enable_frame_2 == "enable" else 0.0
            frame_3_strength = frame_3_strength_low if enable_frame_3 == "enable" else 0.0
            if frame_2_strength == 0.0 or frame_3_strength == 0.0:
                image_low_only = torch.ones((length, height, width, 3), device=device) * 0.5
                if frame_1_image is not None: image_low_only[:frame_1_image.shape[0]] = frame_1_image
                if frame_2_image is not None and frame_2_strength > 0.0: image_low_only[frame_2_idx:frame_2_idx + frame_2_image.shape[0]] = frame_2_image
                if frame_3_image is not None and frame_3_strength > 0.0: image_low_only[frame_3_idx:frame_3_idx + frame_3_image.shape[0]] = frame_3_image
                if frame_4_image is not None: image_low_only[frame_4_idx:frame_4_idx + frame_4_image.shape[0]] = frame_4_image

                concat_latent_image_low = _encode_vae_helper(vae, image_low_only[:, :, :, :3], use_tiled_vae, vae_tile_size, "WanFourFrameRef")
            else:
                concat_latent_image_low = _encode_vae_helper(vae, image[:, :, :, :3], use_tiled_vae, vae_tile_size, "WanFourFrameRef")
        
        mask_high_reshaped = mask_high_noise.view(1, mask_high_noise.shape[2] // 4, 4, mask_high_noise.shape[3], mask_high_noise.shape[4]).transpose(1, 2)
        mask_low_reshaped = mask_low_noise.view(1, mask_low_noise.shape[2] // 4, 4, mask_low_noise.shape[3], mask_low_noise.shape[4]).transpose(1, 2)
        
        positive_high_noise = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image_high, "concat_mask": mask_high_reshaped})
        positive_low_noise = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image_low, "concat_mask": mask_low_reshaped})
        
        clip_vision_output = self._merge_clip_vision_outputs(clip_vision_frame_1, clip_vision_frame_2, clip_vision_frame_3, clip_vision_frame_4)
        if clip_vision_output is not None:
            positive_low_noise = node_helpers.conditioning_set_values(positive_low_noise, {"clip_vision_output": clip_vision_output})
        
        return (positive_high_noise, positive_low_noise, negative, {"samples": latent})

    def _merge_clip_vision_outputs(self, *outputs: Any) -> Optional[Any]:
        valid_outputs = [o for o in outputs if o is not None]
        if not valid_outputs: return None
        if len(valid_outputs) == 1: return valid_outputs[0]
        all_states = [o.penultimate_hidden_states for o in valid_outputs]
        combined_states = torch.cat(all_states, dim=-2)
        result = comfy.clip_vision.Output()
        result.penultimate_hidden_states = combined_states
        return result

class WanAdvancedI2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": MAX_RESOLUTION, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "mode": (["NORMAL", "SINGLE_PERSON"], {"default": "NORMAL"}),
                "start_image": ("IMAGE",),
                "middle_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "middle_frame_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "motion_frames": ("IMAGE",),
                "video_frame_offset": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "long_video_mode": (["DISABLED", "AUTO_CONTINUE", "SVI_SHOT"], {"default": "DISABLED"}),
                "continue_frames_count": ("INT", {"default": 5, "min": 0, "max": 20, "step": 1}),
                "high_noise_mid_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "low_noise_start_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "low_noise_mid_strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "low_noise_end_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "clip_vision_start_image": ("CLIP_VISION_OUTPUT",),
                "clip_vision_middle_image": ("CLIP_VISION_OUTPUT",),
                "clip_vision_end_image": ("CLIP_VISION_OUTPUT",),
                "enable_middle_frame": ("BOOLEAN", {"default": True}),
                "use_tiled_vae": ("BOOLEAN", {"default": False, "label_on": "启用分块编码", "label_off": "禁用分块编码"}),
                "vae_tile_size": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT", "INT", "INT", "INT")
    RETURN_NAMES = ("positive_high_noise", "positive_low_noise", "negative", "latent", "trim_latent", "trim_image", "next_offset")
    FUNCTION = "generate"
    CATEGORY = "📜BaBaAi Tools/Wan22FMLF"

    def generate(self, 
                 positive: Tuple[Any, ...], negative: Tuple[Any, ...], vae: Any,
                 width: int, height: int, length: int, batch_size: int,
                 mode: str = "NORMAL",
                 start_image: Optional[torch.Tensor] = None,
                 middle_image: Optional[torch.Tensor] = None,
                 end_image: Optional[torch.Tensor] = None,
                 middle_frame_ratio: float = 0.5,
                 motion_frames: Optional[torch.Tensor] = None,
                 video_frame_offset: int = 0,
                 long_video_mode: str = "DISABLED",
                 continue_frames_count: int = 5,
                 high_noise_mid_strength: float = 0.8,
                 low_noise_start_strength: float = 1.0,
                 low_noise_mid_strength: float = 0.2,
                 low_noise_end_strength: float = 1.0,
                 clip_vision_start_image: Optional[Any] = None,
                 clip_vision_middle_image: Optional[Any] = None,
                 clip_vision_end_image: Optional[Any] = None,
                 enable_middle_frame: bool = True,
                 use_tiled_vae: bool = False,
                 vae_tile_size: int = 512
                 ) -> Tuple[Tuple[Any, ...], Tuple[Any, ...], Tuple[Any, ...], dict, int, int, int]:
        
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((length - 1) // 4) + 1
        device = comfy.model_management.intermediate_device()
        
        latent = torch.zeros([batch_size, latent_channels, latent_t, height // spacial_scale, width // spacial_scale], device=device)
        
        trim_latent, trim_image, next_offset = 0, 0, 0
        has_motion_frames = (motion_frames is not None and motion_frames.shape[0] > 0)
        is_pure_triple_mode = (not has_motion_frames and long_video_mode == "DISABLED")
        
        if video_frame_offset >= 0:
            if (long_video_mode == "AUTO_CONTINUE" or long_video_mode == "SVI_SHOT") and has_motion_frames and continue_frames_count > 0:
                actual_count = min(continue_frames_count, motion_frames.shape[0])
                motion_frames = motion_frames[-actual_count:]
                video_frame_offset = max(0, video_frame_offset - motion_frames.shape[0])
                trim_image = motion_frames.shape[0]
            if video_frame_offset > 0:
                if start_image is not None and start_image.shape[0] > 1: start_image = start_image[video_frame_offset:] if start_image.shape[0] > video_frame_offset else None
                if middle_image is not None and middle_image.shape[0] > 1: middle_image = middle_image[video_frame_offset:] if middle_image.shape[0] > video_frame_offset else None
                if end_image is not None and end_image.shape[0] > 1: end_image = end_image[video_frame_offset:] if end_image.shape[0] > video_frame_offset else None
            next_offset = video_frame_offset + length
        
        if motion_frames is not None: motion_frames = comfy.utils.common_upscale(motion_frames.movedim(-1, 1), width, height, "area", "center").movedim(1, -1)
        if start_image is not None: start_image = comfy.utils.common_upscale(start_image[:length if is_pure_triple_mode else 1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if middle_image is not None: middle_image = comfy.utils.common_upscale(middle_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if end_image is not None: end_image = comfy.utils.common_upscale(end_image[-length if is_pure_triple_mode else -1:].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        
        image = torch.ones((length, height, width, 3), device=device) * 0.5
        mask_base = torch.ones((1, 1, latent_t * 4, latent.shape[-2], latent.shape[-1]), device=device)
        
        middle_idx = self._calculate_aligned_position(middle_frame_ratio, length)[0]
        middle_idx = max(4, min(middle_idx, length - 5))
        
        mask_high_noise = mask_base.clone()
        mask_low_noise = mask_base.clone()
        svi_shot_second_pass = False
        
        if long_video_mode == "SVI_SHOT" and start_image is not None:
            if has_motion_frames: svi_shot_second_pass = True
            else:
                image[:start_image.shape[0]] = start_image[..., :3]
                start_latent_frames = ((start_image.shape[0] - 1) // 4) + 1
                mask_high_noise[:, :, :start_latent_frames * 4] = 0.0
                mask_low_noise[:, :, :start_latent_frames * 4] = max(0.0, 1.0 - low_noise_start_strength)
        
        if has_motion_frames and not (long_video_mode == "SVI_SHOT" and not svi_shot_second_pass):
            image[:motion_frames.shape[0]] = motion_frames[..., :3]
            motion_latent_frames = ((motion_frames.shape[0] - 1) // 4) + 1
            mask_high_noise[:, :, :motion_latent_frames * 4] = 0.0
            if not svi_shot_second_pass: mask_low_noise[:, :, :motion_latent_frames * 4] = 0.0
            
            if middle_image is not None and enable_middle_frame:
                image[middle_idx:middle_idx + 1] = middle_image
                start_range, end_range = max(0, middle_idx), min(length, middle_idx + 4)
                mask_high_noise[:, :, start_range:end_range] = max(0.0, 1.0 - high_noise_mid_strength)
                mask_low_noise[:, :, start_range:end_range] = max(0.0, 1.0 - low_noise_mid_strength)
            
            if end_image is not None:
                image[-1:] = end_image[..., :3]
                mask_high_noise[:, :, -1:] = 0.0
                mask_low_noise[:, :, -1:] = max(0.0, 1.0 - low_noise_end_strength)
        else:
            if start_image is not None:
                image[:start_image.shape[0]] = start_image[..., :3]
                if is_pure_triple_mode:
                    mask_range = min(start_image.shape[0] + 3, length)
                    mask_high_noise[:, :, :mask_range] = 0.0
                    mask_low_noise[:, :, :mask_range] = max(0.0, 1.0 - low_noise_start_strength)
                else:
                    start_latent_frames = ((start_image.shape[0] - 1) // 4) + 1
                    mask_high_noise[:, :, :start_latent_frames * 4] = 0.0
                    mask_low_noise[:, :, :start_latent_frames * 4] = max(0.0, 1.0 - low_noise_start_strength)
            
            if middle_image is not None and enable_middle_frame:
                image[middle_idx:middle_idx + 1] = middle_image
                start_range, end_range = max(0, middle_idx), min(length, middle_idx + 4)
                mask_high_noise[:, :, start_range:end_range] = max(0.0, 1.0 - high_noise_mid_strength)
                mask_low_noise[:, :, start_range:end_range] = max(0.0, 1.0 - low_noise_mid_strength)
            
            if end_image is not None:
                image[-end_image.shape[0]:] = end_image[..., :3]
                if is_pure_triple_mode:
                    mask_high_noise[:, :, -end_image.shape[0]:] = 0.0
                    mask_low_noise[:, :, -end_image.shape[0]:] = max(0.0, 1.0 - low_noise_end_strength)
                else:
                    mask_high_noise[:, :, -1:] = 0.0
                    mask_low_noise[:, :, -1:] = max(0.0, 1.0 - low_noise_end_strength)
        
        concat_latent_image = _encode_vae_helper(vae, image[:, :, :, :3], use_tiled_vae, vae_tile_size, "WanAdvancedI2V")
        
        if svi_shot_second_pass:
            image_low = torch.ones((length, height, width, 3), device=device) * 0.5
            if mode == "SINGLE_PERSON":
                if start_image is not None:
                    image_low[0] = start_image[0, ..., :3]
                    mask_low_noise[:, :, 0:4] = 0.0
            else:
                if motion_frames is not None:
                    image_low[0] = motion_frames[0, ..., :3]
                    mask_low_noise[:, :, 0:1] = 0.0

            concat_latent_image_low = _encode_vae_helper(vae, image_low[:, :, :, :3], use_tiled_vae, vae_tile_size, "WanAdvancedI2V")
        elif mode == "SINGLE_PERSON":
            image_low_only = torch.ones((length, height, width, 3), device=device) * 0.5
            if motion_frames is not None: image_low_only[:motion_frames.shape[0]] = motion_frames[..., :3]
            elif start_image is not None: image_low_only[:start_image.shape[0]] = start_image[..., :3]

            concat_latent_image_low = _encode_vae_helper(vae, image_low_only[:, :, :, :3], use_tiled_vae, vae_tile_size, "WanAdvancedI2V")
        elif low_noise_start_strength == 0.0 or low_noise_mid_strength == 0.0 or low_noise_end_strength == 0.0:
            image_low_only = torch.ones((length, height, width, 3), device=device) * 0.5
            if motion_frames is not None and low_noise_start_strength > 0.0: image_low_only[:motion_frames.shape[0]] = motion_frames[..., :3]
            elif start_image is not None and low_noise_start_strength > 0.0: image_low_only[:start_image.shape[0]] = start_image[..., :3]
            if middle_image is not None and low_noise_mid_strength > 0.0 and enable_middle_frame: image_low_only[middle_idx:middle_idx + 1] = middle_image
            if end_image is not None and low_noise_end_strength > 0.0:
                image_low_only[-end_image.shape[0] if is_pure_triple_mode else -1:] = end_image[..., :3]

            concat_latent_image_low = _encode_vae_helper(vae, image_low_only[:, :, :, :3], use_tiled_vae, vae_tile_size, "WanAdvancedI2V")
        else:
            concat_latent_image_low = concat_latent_image
        
        mask_high_reshaped = mask_high_noise.view(1, mask_high_noise.shape[2] // 4, 4, mask_high_noise.shape[3], mask_high_noise.shape[4]).transpose(1, 2)
        mask_low_reshaped = mask_low_noise.view(1, mask_low_noise.shape[2] // 4, 4, mask_low_noise.shape[3], mask_low_noise.shape[4]).transpose(1, 2)
        
        positive_high_noise = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask_high_reshaped})
        positive_low_noise = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image_low, "concat_mask": mask_low_reshaped})
        negative_out = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask_high_reshaped})
        
        clip_vision_output = self._merge_clip_vision_outputs(clip_vision_start_image, clip_vision_middle_image, clip_vision_end_image)
        if clip_vision_output is not None:
            positive_low_noise = node_helpers.conditioning_set_values(positive_low_noise, {"clip_vision_output": clip_vision_output})
            negative_out = node_helpers.conditioning_set_values(negative_out, {"clip_vision_output": clip_vision_output})
        
        return (positive_high_noise, positive_low_noise, negative_out, {"samples": latent}, trim_latent, trim_image, next_offset)
    
    def _calculate_aligned_position(self, ratio: float, total_frames: int) -> Tuple[int, int]:
        desired_pixel_idx = int(total_frames * ratio)
        latent_idx = desired_pixel_idx // 4
        aligned_pixel_idx = latent_idx * 4
        return max(0, min(aligned_pixel_idx, total_frames - 1)), latent_idx
    
    def _merge_clip_vision_outputs(self, *outputs: Any) -> Optional[Any]:
        valid_outputs = [o for o in outputs if o is not None]
        if not valid_outputs: return None
        if len(valid_outputs) == 1: return valid_outputs[0]
        all_states = [o.penultimate_hidden_states for o in valid_outputs]
        combined_states = torch.cat(all_states, dim=-2)
        result = comfy.clip_vision.Output()
        result.penultimate_hidden_states = combined_states
        return result

class WanAdvancedExtractLastFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "num_frames": ("INT", {"default": 9, "min": 0, "max": 81, "step": 1}),}}
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("last_frames",)
    FUNCTION = "extract"
    CATEGORY = "📜BaBaAi Tools/Wan22FMLF"
    def extract(self, samples: dict, num_frames: int) -> Tuple[dict]:
        if num_frames == 0: return ({"samples": torch.zeros_like(samples["samples"][:, :, :0])},)
        latent_frames = ((num_frames - 1) // 4) + 1
        return ({"samples": samples["samples"][:, :, -latent_frames:].clone()},)

class WanAdvancedExtractLastImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "num_frames": ("INT", {"default": 9, "min": 0, "max": 81, "step": 1}),}}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("last_images",)
    FUNCTION = "extract"
    CATEGORY = "📜BaBaAi Tools/Wan22FMLF"
    def extract(self, images: torch.Tensor, num_frames: int) -> Tuple[torch.Tensor]:
        if num_frames == 0: return (images[:0].clone(),)
        return (images[-num_frames:].clone(),)

NODE_CLASS_MAPPINGS = {
    "WanFirstMiddleLastFrameToVideo_Tiled": WanFirstMiddleLastFrameToVideo,
    "WanMultiFrameRefToVideo_Tiled": WanMultiFrameRefToVideo,
    "WanFourFrameReferenceUltimate_Tiled": WanFourFrameReferenceUltimate,
    "WanAdvancedI2V_Tiled": WanAdvancedI2V,
    "WanAdvancedExtractLastFrames": WanAdvancedExtractLastFrames,
    "WanAdvancedExtractLastImages": WanAdvancedExtractLastImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanFirstMiddleLastFrameToVideo_Tiled": "Wan First-Middle-Last Frame (Tiled)",
    "WanMultiFrameRefToVideo_Tiled": "Wan Multi-Frame Reference (Tiled)",
    "WanFourFrameReferenceUltimate_Tiled": "Wan 4-Frame Reference (Tiled)",
    "WanAdvancedI2V_Tiled": "Wan Advanced I2V (Ultimate, Tiled)",
    "WanAdvancedExtractLastFrames": "Wan Extract Last Frames (Latent)",
    "WanAdvancedExtractLastImages": "Wan Extract Last Images",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']