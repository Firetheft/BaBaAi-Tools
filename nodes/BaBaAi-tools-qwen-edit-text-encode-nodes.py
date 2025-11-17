import node_helpers
import comfy.utils
import math
import torch
import nodes  
import inspect  
import logging  

class TextEncodeQwenImageEditPlus_BaBaAi:
    upscale_methods = ["lanczos", "bicubic", "area"]
    crop_methods = ["disabled", "center"]
    resolution_options = [512, 768, 1024, 1344, 1536, 2048]
    divisor_options = [8, 16, 32, 64]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "clip": ("CLIP", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": 
            {
                "vae": ("VAE", ),
                "image1": ("IMAGE", ),
                "mask1": ("MASK", ),
                "image2": ("IMAGE", ),
                "mask2": ("MASK", ),
                "image3": ("IMAGE", ),
                "mask3": ("MASK", ),
                "image4": ("IMAGE", ),
                "mask4": ("MASK", ),
                "image5": ("IMAGE", ),
                "mask5": ("MASK", ),
                "enable_resize": ("BOOLEAN", {"default": True}),
                "enable_vl_resize": ("BOOLEAN", {"default": True}),
                "skip_first_image_resize": ("BOOLEAN", {"default": False}),
                "upscale_method": (s.upscale_methods,),
                "crop": (s.crop_methods,),
                "resolution_side": (s.resolution_options, {"default": 1024}),
                "divisible_by": (s.divisor_options, {"default": 8}),
                "use_tiled_vae": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "禁用"}), 
                "vae_tile_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}), 
                "instruction": ("STRING", {"multiline": True, "default": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."}),
                
            }
        }

    RETURN_TYPES = ("CONDITIONING", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "LATENT", )
    RETURN_NAMES = ("conditioning", "image1", "image2", "image3", "image4", "image5", "latent")
    FUNCTION = "encode"

    CATEGORY = "📜BaBaAi Tools/QwenImageEdit文本编码器"

    def encode(self, clip, prompt, vae=None, 
               image1=None, mask1=None, image2=None, mask2=None, image3=None, mask3=None, image4=None, mask4=None, image5=None, mask5=None, 
               enable_resize=True, enable_vl_resize=True, skip_first_image_resize=False,
               upscale_method="bicubic",
               crop="center",
               resolution_side=1024,
               divisible_by=16,
               instruction="",
               use_tiled_vae=False, 
               vae_tile_size=512     
               ):
        
        div_float = float(divisible_by)
        ref_latents = []
        ref_masks = []
        
        images = [image1, image2, image3, image4, image5]
        masks = [mask1, mask2, mask3, mask4, mask5]
        
        images_vl = []
        vae_images = []
        template_prefix = "<|im_start|>system\n"
        template_suffix = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        instruction_content = ""
        if instruction == "":
            instruction_content = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
        else:
            if template_prefix in instruction:
                instruction = instruction.split(template_prefix)[1]
            if template_suffix in instruction:
                instruction = instruction.split(template_suffix)[0]
            if "{}" in instruction:
                instruction = instruction.replace("{}", "")
            instruction_content = instruction
        llama_template = template_prefix + instruction_content + template_suffix
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                mask = masks[i]
                current_noise_mask = None
                
                samples = image.movedim(-1, 1)
                _, c, _, _ = samples.shape
                
                mask_samples = None
                if mask is not None:
                    mask_samples = mask.unsqueeze(1)
                    if samples.shape[2:] != mask_samples.shape[2:]:
                        print(f"[BaBaAi-QwenEdit] 警告: 图像 {i+1} 和 蒙版 {i+1} 的尺寸不匹配。蒙版将被忽略。")
                        mask_samples = None
                        mask = None
                    else:
                        mask_samples = mask_samples.repeat(1, c, 1, 1)
                
                current_total = (samples.shape[3] * samples.shape[2])
                total = int(resolution_side * resolution_side)
                scale_by = 1
                if enable_resize:
                    scale_by = math.sqrt(total / current_total)
                width = round(samples.shape[3] * scale_by / div_float) * divisible_by
                height = round(samples.shape[2] * scale_by / div_float) * divisible_by
                if vae is not None:
                    s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                    
                    if mask is not None and mask_samples is not None:
                        m = comfy.utils.common_upscale(mask_samples, width, height, upscale_method, crop)
                        current_noise_mask = m[:, :1, :, :].squeeze(1)
                        
                    image = s.movedim(1, -1)
                    
                    pixels_for_vae = image[:, :, :, :3]
                    if use_tiled_vae:
                        try:
                            encoder = nodes.VAEEncodeTiled()
                            if 'overlap' in inspect.signature(encoder.encode).parameters:
                                encoded_latent = encoder.encode(vae, pixels_for_vae, vae_tile_size, overlap=64)[0]
                            else:
                                logging.warning("[BaBaAi-QwenEdit] 您的ComfyUI版本较旧。Tiled VAE overlap功能不可用。")
                                encoded_latent = encoder.encode(vae, pixels_for_vae, vae_tile_size)[0]
                        except Exception as e:
                            print(f"[BaBaAi-QwenEdit] VAE分块编码失败，回退到标准编码。错误: {e}")
                            encoded_latent = nodes.VAEEncode().encode(vae, pixels_for_vae)[0] 
                    else:
                        encoded_latent = nodes.VAEEncode().encode(vae, pixels_for_vae)[0] 
                    
                    ref_latents.append(encoded_latent["samples"])
                    ref_masks.append(current_noise_mask)
                    
                    vae_images.append(image)
                    
                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
                if enable_vl_resize and not skip_first_image_resize and i == 0:
                    total = int(384 * 384)
                    scale_by = math.sqrt(total / current_total)
                    width = round(samples.shape[3] * scale_by / div_float) * divisible_by
                    height = round(samples.shape[2] * scale_by / div_float) * divisible_by

                s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                image = s.movedim(1, -1)
                images_vl.append(image)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)

        final_noise_mask = None
        if len(ref_latents) > 0:
            samples = ref_latents[0]
            final_noise_mask = ref_masks[0]
        else:
            samples = torch.zeros(1, 4, 128, 128)

        latent_out = {"samples": samples}
        if final_noise_mask is not None:
            latent_out["noise_mask"] = final_noise_mask
            
        if len(vae_images) < 5:
            vae_images.extend([None] * (5 - len(vae_images)))
        o_image1, o_image2, o_image3, o_image4, o_image5 = vae_images
        return (conditioning, o_image1, o_image2, o_image3, o_image4, o_image5, latent_out)

class TextEncodeQwenImageEditPlusAdvance_BaBaAi:
    upscale_methods = ["lanczos", "bicubic", "area"]
    crop_methods = ["pad", "center", "disabled"]
    target_sizes = [1024, 1344, 1536, 2048, 768, 512]
    target_vl_sizes = [392,384]
    divisor_options = [8, 16, 32, 64]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "clip": ("CLIP", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": 
            {
                "vae": ("VAE", ),
                "vl_resize_image1": ("IMAGE", ),
                "vl_resize_mask1": ("MASK", ),
                "vl_resize_image2": ("IMAGE", ),
                "vl_resize_mask2": ("MASK", ),
                "vl_resize_image3": ("IMAGE", ),
                "vl_resize_mask3": ("MASK", ),
                "not_resize_image1": ("IMAGE", ),
                "not_resize_mask1": ("MASK", ),
                "not_resize_image2": ("IMAGE", ),
                "not_resize_mask2": ("MASK", ),
                "not_resize_image3": ("IMAGE", ),
                "not_resize_mask3": ("MASK", ),
                "target_size": (s.target_sizes, {"default": 1024}),
                "target_vl_size": (s.target_vl_sizes, {"default": 384}),
                "upscale_method": (s.upscale_methods,),
                "crop_method": (s.crop_methods,),
                "divisible_by": (s.divisor_options, {"default": 8}),
                "use_tiled_vae": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "禁用"}), 
                "vae_tile_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}), 
                "instruction": ("STRING", {"multiline": True, "default": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "CONDITIONING", "ANY")
    RETURN_NAMES = ("conditioning_with_full_ref", "latent", "target_image1", "target_image2", "target_image3", "vl_resized_image1", "vl_resized_image2", "vl_resized_image3", "conditioning_with_first_ref", "pad_info")
    FUNCTION = "encode"

    CATEGORY = "📜BaBaAi Tools/QwenImageEdit文本编码器"

    def encode(self, clip, prompt, vae=None, 
               vl_resize_image1=None, vl_resize_mask1=None, vl_resize_image2=None, vl_resize_mask2=None, vl_resize_image3=None, vl_resize_mask3=None,
               not_resize_image1=None, not_resize_mask1=None, not_resize_image2=None, not_resize_mask2=None, not_resize_image3=None, not_resize_mask3=None, 
               target_size=1024, 
               target_vl_size=384,
               upscale_method="lanczos",
               crop_method="pad",
               divisible_by=8,
               instruction="",
               use_tiled_vae=False, 
               vae_tile_size=512     
               ):
        
        div_float = float(divisible_by)
        pad_info = {
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0,
            "scale_by": 0
        }
        ref_latents = []
        ref_masks = [] 
        ref_pad_infos = []

        images = [
            {
                "image": not_resize_image1,
                "mask": not_resize_mask1,
                "vl_resize": False 
            },
            {
                "image": not_resize_image2,
                "mask": not_resize_mask2,
                "vl_resize": False 
            },
            {
                "image": not_resize_image3,
                "mask": not_resize_mask3,
                "vl_resize": False 
            },
            {
                "image": vl_resize_image1,
                "mask": vl_resize_mask1,
                "vl_resize": True 
            },
            {
                "image": vl_resize_image2,
                "mask": vl_resize_mask2,
                "vl_resize": True 
            },
            {
                "image": vl_resize_image3,
                "mask": vl_resize_mask3,
                "vl_resize": True 
            }
        ]
        
        vl_resized_images = []
        vae_images = []
        vl_images = []
        template_prefix = "<|im_start|>system\n"
        template_suffix = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        instruction_content = ""
        if instruction == "":
            instruction_content = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
        else:
            if template_prefix in instruction:
                instruction = instruction.split(template_prefix)[1]
            if template_suffix in instruction:
                instruction = instruction.split(template_suffix)[0]
            if "{}" in instruction:
                instruction = instruction.replace("{}", "")
            instruction_content = instruction
        llama_template = template_prefix + instruction_content + template_suffix
        image_prompt = ""

        if vae is not None:
            for i, image_obj in enumerate(images):
                image = image_obj["image"]
                mask = image_obj["mask"] 
                vl_resize = image_obj["vl_resize"]
                current_noise_mask = None 
                current_pad_info = None
                
                if image is not None:
                    samples = image.movedim(-1, 1)
                    _, c, _, _ = samples.shape
                    
                    mask_samples = None
                    if mask is not None:
                        mask_samples = mask.unsqueeze(1) 
                        if samples.shape[2:] != mask_samples.shape[2:]:
                            print(f"[BaBaAi-QwenEdit] 警告: 图像 {i+1} 和 蒙版 {i+1} 的尺寸不匹配。蒙版将被忽略。")
                            mask_samples = None
                            mask = None
                        else:
                            mask_samples = mask_samples.repeat(1, c, 1, 1)

                    current_total = (samples.shape[3] * samples.shape[2])
                    total = int(target_size * target_size)
                    scale_by = math.sqrt(total / current_total)
                    if crop_method == "pad":
                        crop = "center"
                        scaled_width = round(samples.shape[3] * scale_by)
                        scaled_height = round(samples.shape[2] * scale_by)
                        canvas_width = math.ceil(samples.shape[3] * scale_by / div_float) * divisible_by
                        canvas_height = math.ceil(samples.shape[2] * scale_by / div_float) * divisible_by
                        
                        canvas = torch.zeros(
                            (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                            dtype=samples.dtype,
                            device=samples.device
                        )
                        resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, upscale_method, crop)
                        resized_width = resized_samples.shape[3]
                        resized_height = resized_samples.shape[2]
                        
                        canvas[:, :, :resized_height, :resized_width] = resized_samples
                        
                        if mask is not None and mask_samples is not None:
                            mask_canvas = torch.zeros(
                                (samples.shape[0], c, canvas_height, canvas_width),
                                dtype=samples.dtype,
                                device=samples.device
                            )
                            resized_mask_samples = comfy.utils.common_upscale(mask_samples, scaled_width, scaled_height, upscale_method, crop)
                            mask_canvas[:, :, :resized_height, :resized_width] = resized_mask_samples
                            current_noise_mask = mask_canvas[:, :1, :, :].squeeze(1)
                            
                        actual_resized_total = int(resized_width * resized_height)
                        if current_total == 0:
                            actual_scale_by = 1.0
                        else:
                            actual_scale_by = math.sqrt(actual_resized_total / current_total)

                        current_pad_info = {
                            "x": 0,
                            "y": 0,
                            "width": canvas_width - resized_width,
                            "height": canvas_height - resized_height,
                            "scale_by": round(1 / actual_scale_by, 5) if actual_scale_by != 0 else 1.0
                        }
                        
                        s = canvas
                    else:
                        width = round(samples.shape[3] * scale_by / div_float) * divisible_by
                        height = round(samples.shape[2] * scale_by / div_float) * divisible_by
                        crop = crop_method
                        s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                        
                        if mask is not None and mask_samples is not None:
                            m = comfy.utils.common_upscale(mask_samples, width, height, upscale_method, crop)
                            current_noise_mask = m[:, :1, :, :].squeeze(1)

                    image = s.movedim(1, -1)
                    
                    pixels_for_vae = image[:, :, :, :3]
                    if use_tiled_vae:
                        try:
                            encoder = nodes.VAEEncodeTiled()
                            if 'overlap' in inspect.signature(encoder.encode).parameters:
                                encoded_latent = encoder.encode(vae, pixels_for_vae, vae_tile_size, overlap=64)[0]
                            else:
                                logging.warning("[BaBaAi-QwenEdit] 您的ComfyUI版本较旧。Tiled VAE overlap功能不可用。")
                                encoded_latent = encoder.encode(vae, pixels_for_vae, vae_tile_size)[0]
                        except Exception as e:
                            print(f"[BaBaAi-QwenEdit] VAE分块编码失败，回退到标准编码。错误: {e}")
                            encoded_latent = nodes.VAEEncode().encode(vae, pixels_for_vae)[0] 
                    else:
                        encoded_latent = nodes.VAEEncode().encode(vae, pixels_for_vae)[0] 
                    
                    ref_latents.append(encoded_latent["samples"])
                    ref_masks.append(current_noise_mask) 
                    ref_pad_infos.append(current_pad_info)
                    
                    vae_images.append(image)
                    
                    if vl_resize:
                        total = int(target_vl_size * target_vl_size)
                        scale_by = math.sqrt(total / current_total)
                        
                        if crop_method == "pad":
                            crop = "center"
                            scaled_width = round(samples.shape[3] * scale_by)
                            scaled_height = round(samples.shape[2] * scale_by)
                            canvas_width = math.ceil(samples.shape[3] * scale_by / div_float) * divisible_by
                            canvas_height = math.ceil(samples.shape[2] * scale_by / div_float) * divisible_by
                            
                            canvas = torch.zeros(
                                (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                                dtype=samples.dtype,
                                device=samples.device
                            )
                            resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, upscale_method, crop)
                            resized_width = resized_samples.shape[3]
                            resized_height = resized_samples.shape[2]
                            
                            canvas[:, :, :resized_height, :resized_width] = resized_samples
                            s = canvas
                        else:
                            width = round(samples.shape[3] * scale_by / div_float) * divisible_by
                            height = round(samples.shape[2] * scale_by / div_float) * divisible_by
                            crop = crop_method
                            s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                        
                        image = s.movedim(1, -1)
                        vl_resized_images.append(image)

                    image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
                    vl_images.append(image)
                    
                
        tokens = clip.tokenize(image_prompt + prompt, images=vl_images, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning_full_ref = conditioning
        conditioning_with_first_ref = conditioning
        if len(ref_latents) > 0:
            conditioning_full_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)          
            conditioning_with_first_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latents[0]]}, append=True)

        final_noise_mask = None
        if len(ref_latents) > 0:
            samples = ref_latents[0]
            final_noise_mask = ref_masks[0] 
            
            final_pad_info = ref_pad_infos[0]
            if final_pad_info is not None:
                pad_info = final_pad_info
            
        else:
            samples = torch.zeros(1, 4, 128, 128)

        latent_out = {"samples": samples}
        if final_noise_mask is not None:
            latent_out["noise_mask"] = final_noise_mask 
            
        if len(vae_images) < 3:
            vae_images.extend([None] * (3 - len(vae_images)))
        o_image1, o_image2, o_image3 = vae_images
        
        if len(vl_resized_images) < 3:
            vl_resized_images.extend([None] * (3 - len(vl_resized_images)))
        vl_image1, vl_image2, vl_image3 = vl_resized_images
        
        return (conditioning_full_ref, latent_out, o_image1, o_image2, o_image3, vl_image1, vl_image2, vl_image3, conditioning_with_first_ref, pad_info)

def validate_vl_resize_indexs(vl_resize_indexs_str, valid_length):
    try:
        indexes = [int(i)-1 for i in vl_resize_indexs_str.split(",")]
        indexes = list(set(indexes))
    except ValueError as e:
        raise ValueError(f"Invalid format for vl_resize_indexs: {e}")

    if not indexes:
        raise ValueError("vl_resize_indexs must not be empty")

    indexes = [idx for idx in indexes if 0 <= idx < valid_length]

    return indexes

class TextEncodeQwenImageEditPlusPro_BaBaAi:
    upscale_methods = ["lanczos", "bicubic", "area"]
    crop_methods = ["pad", "center", "disabled"]
    target_sizes = [1024, 1344, 1536, 2048, 768, 512]
    target_vl_sizes = [392,384]
    vl_resize_indexs = [1,2,3]
    main_image_index = 1
    divisor_options = [8, 16, 32, 64]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "clip": ("CLIP", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": 
            {
                "vae": ("VAE", ),
                "image1": ("IMAGE", ),
                "mask1": ("MASK", ),
                "image2": ("IMAGE", ),
                "mask2": ("MASK", ),
                "image3": ("IMAGE", ),
                "mask3": ("MASK", ),
                "image4": ("IMAGE", ),
                "mask4": ("MASK", ),
                "image5": ("IMAGE", ),
                "mask5": ("MASK", ),
                "vl_resize_indexs": ("STRING", {"default": "1,2,3"}),
                "main_image_index": ("INT", {"default": 1, "max": 5, "min": 1}),
                "target_size": (s.target_sizes, {"default": 1024}),
                "target_vl_size": (s.target_vl_sizes, {"default": 384}),
                "upscale_method": (s.upscale_methods,),
                "crop_method": (s.crop_methods,),
                "divisible_by": (s.divisor_options, {"default": 8}),
                "use_tiled_vae": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "禁用"}), 
                "vae_tile_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}), 
                "instruction": ("STRING", {"multiline": True, "default": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."}),
                
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "CONDITIONING", "ANY")
    RETURN_NAMES = ("conditioning_with_full_ref", "latent", "image1", "image2", "image3", "image4", "image5", "conditioning_with_main_ref", "pad_info")
    FUNCTION = "encode"

    CATEGORY = "📜BaBaAi Tools/QwenImageEdit文本编码器"

    def encode(self, clip, prompt, vae=None, 
               image1=None, mask1=None, image2=None, mask2=None, image3=None, mask3=None,
               image4=None, mask4=None, image5=None, mask5=None, 
               vl_resize_indexs="1,2,3",
               main_image_index=1,
               target_size=1024, 
               target_vl_size=384,
               upscale_method="lanczos",
               crop_method="pad",
               divisible_by=8,
               instruction="",
               use_tiled_vae=False, 
               vae_tile_size=512     
               ):
        
        div_float = float(divisible_by)
        resize_indexs = validate_vl_resize_indexs(vl_resize_indexs,5)
        pad_info = {
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0,
            "scale_by": 0
        }
        ref_latents = []
        ref_masks = []
        
        temp = [image1, image2, image3, image4, image5]
        masks_temp = [mask1, mask2, mask3, mask4, mask5] 
        
        images = []
        for i, image in enumerate(temp):
            image_dict = {
                "image": image,
                "mask": masks_temp[i],
                "vl_resize": False
            }
            if i in resize_indexs:
                image_dict['vl_resize'] = True
            images.append(image_dict)
            
        vl_resized_images = []
        
        vae_images = []
        vl_images = []
        template_prefix = "<|im_start|>system\n"
        template_suffix = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        instruction_content = ""
        if instruction == "":
            instruction_content = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
        else:
            if template_prefix in instruction:
                instruction = instruction.split(template_prefix)[1]
            if template_suffix in instruction:
                instruction = instruction.split(template_suffix)[0]
            if "{}" in instruction:
                instruction = instruction.replace("{}", "")
            instruction_content = instruction
        llama_template = template_prefix + instruction_content + template_suffix
        image_prompt = ""
        
        main_image_idx_user = main_image_index - 1 

        if vae is not None:
            for i, image_obj in enumerate(images):
                image = image_obj["image"]
                mask = image_obj["mask"]
                vl_resize = image_obj["vl_resize"]
                current_noise_mask = None 
                
                if image is not None:
                    samples = image.movedim(-1, 1)
                    _, c, _, _ = samples.shape
                    
                    mask_samples = None
                    if mask is not None:
                        mask_samples = mask.unsqueeze(1) 
                        if samples.shape[2:] != mask_samples.shape[2:]:
                            print(f"[BaBaAi-QwenEdit] 警告: 图像 {i+1} 和 蒙版 {i+1} 的尺寸不匹配。蒙版将被忽略。")
                            mask_samples = None
                            mask = None
                        else:
                            mask_samples = mask_samples.repeat(1, c, 1, 1)

                    current_total = (samples.shape[3] * samples.shape[2])
                    total = int(target_size * target_size)
                    scale_by = math.sqrt(total / current_total)
                    if crop_method == "pad":
                        crop = "center"
                        scaled_width = round(samples.shape[3] * scale_by)
                        scaled_height = round(samples.shape[2] * scale_by)
                        canvas_width = math.ceil(samples.shape[3] * scale_by / div_float) * divisible_by
                        canvas_height = math.ceil(samples.shape[2] * scale_by / div_float) * divisible_by
                        
                        canvas = torch.zeros(
                            (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                            dtype=samples.dtype,
                            device=samples.device
                        )
                        resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, upscale_method, crop)
                        resized_width = resized_samples.shape[3]
                        resized_height = resized_samples.shape[2]
                        
                        canvas[:, :, :resized_height, :resized_width] = resized_samples
                        
                        if mask is not None and mask_samples is not None:
                            mask_canvas = torch.zeros(
                                (samples.shape[0], c, canvas_height, canvas_width), 
                                dtype=samples.dtype,
                                device=samples.device
                            )
                            resized_mask_samples = comfy.utils.common_upscale(mask_samples, scaled_width, scaled_height, upscale_method, crop)
                            mask_canvas[:, :, :resized_height, :resized_width] = resized_mask_samples
                            current_noise_mask = mask_canvas[:, :1, :, :].squeeze(1)

                        if i == main_image_idx_user:
                            actual_resized_total = int(resized_width * resized_height)
                            if current_total == 0:
                                actual_scale_by = 1.0
                            else:
                                actual_scale_by = math.sqrt(actual_resized_total / current_total)

                            pad_info = {
                                "x": 0,
                                "y": 0,
                                "width": canvas_width - resized_width,
                                "height": canvas_height - resized_height,
                                "scale_by": round(1 / actual_scale_by, 5) if actual_scale_by != 0 else 1.0
                            }

                        s = canvas
                    else:
                        width = round(samples.shape[3] * scale_by / div_float) * divisible_by
                        height = round(samples.shape[2] * scale_by / div_float) * divisible_by
                        crop = crop_method
                        s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                        
                        if mask is not None and mask_samples is not None:
                            m = comfy.utils.common_upscale(mask_samples, width, height, upscale_method, crop)
                            current_noise_mask = m[:, :1, :, :].squeeze(1)
                        
                    image = s.movedim(1, -1)
                    
                    pixels_for_vae = image[:, :, :, :3]
                    if use_tiled_vae:
                        try:
                            encoder = nodes.VAEEncodeTiled()
                            if 'overlap' in inspect.signature(encoder.encode).parameters:
                                encoded_latent = encoder.encode(vae, pixels_for_vae, vae_tile_size, overlap=64)[0]
                            else:
                                logging.warning("[BaBaAi-QwenEdit] 您的ComfyUI版本较旧。Tiled VAE overlap功能不可用。")
                                encoded_latent = encoder.encode(vae, pixels_for_vae, vae_tile_size)[0]
                        except Exception as e:
                            print(f"[BaBaAi-QwenEdit] VAE分块编码失败，回退到标准编码。错误: {e}")
                            encoded_latent = nodes.VAEEncode().encode(vae, pixels_for_vae)[0] 
                    else:
                        encoded_latent = nodes.VAEEncode().encode(vae, pixels_for_vae)[0] 
                    
                    ref_latents.append(encoded_latent["samples"])
                    ref_masks.append(current_noise_mask) 
                    
                    vae_images.append(image)
                    
                    if vl_resize:
                        total = int(target_vl_size * target_vl_size)
                        scale_by = math.sqrt(total / current_total)
                        
                        if crop_method == "pad":
                            crop = "center"
                            scaled_width = round(samples.shape[3] * scale_by)
                            scaled_height = round(samples.shape[2] * scale_by)
                            canvas_width = math.ceil(samples.shape[3] * scale_by / div_float) * divisible_by
                            canvas_height = math.ceil(samples.shape[2] * scale_by / div_float) * divisible_by
                            
                            canvas = torch.zeros(
                                (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                                dtype=samples.dtype,
                                device=samples.device
                            )
                            resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, upscale_method, crop)
                            resized_width = resized_samples.shape[3]
                            resized_height = resized_samples.shape[2]
                            
                            canvas[:, :, :resized_height, :resized_width] = resized_samples
                            s = canvas
                        else:
                            width = round(samples.shape[3] * scale_by / div_float) * divisible_by
                            height = round(samples.shape[2] * scale_by / div_float) * divisible_by
                            crop = crop_method
                            s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                        
                        image = s.movedim(1, -1)
                        vl_resized_images.append(image)

                    image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
                    vl_images.append(image)
                    
                
        tokens = clip.tokenize(image_prompt + prompt, images=vl_images, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning_full_ref = conditioning
        conditioning_with_main_ref = conditioning
        
        samples = torch.zeros(1, 4, 128, 128)
        final_noise_mask = None 
        
        if len(ref_latents) > 0:
            main_image_idx = main_image_index - 1
            if main_image_idx < 0 or main_image_idx >= len(ref_latents): 
                print("\n 自动修复 main_image_index 到第一个图像索引")
                main_image_idx = 0
                
            conditioning_full_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
            conditioning_with_main_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latents[main_image_idx]]}, append=True)
            samples = ref_latents[main_image_idx]
            final_noise_mask = ref_masks[main_image_idx]
        
        latent_out = {"samples": samples}
        if final_noise_mask is not None:
            latent_out["noise_mask"] = final_noise_mask
            
        if len(vae_images) < len(images):
            vae_images.extend([None] * (len(images) - len(vae_images)))
        image1, image2, image3, image4, image5 = vae_images
        
        return (conditioning_full_ref, latent_out, image1, image2, image3, image4, image5, conditioning_with_main_ref, pad_info)

class CropWithPadInfo_BaBaAi:
    upscale_methods = ["lanczos", "bicubic", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "pad_info": ("ANY", ),
                
                "enable_scaling": ("BOOLEAN", {
                    "default": True, 
                    "label_on": "启用缩放", 
                    "label_off": "仅裁切"
                }),
                
                "latent_upscale_factor": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.1, 
                    "max": 16.0,
                    "step": 0.1,
                    "round": 0.01,
                    "display": "number"
                }),
                
                "scale_method": (s.upscale_methods, {"default": "lanczos"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",) 
    FUNCTION = "crop_and_scale"
    CATEGORY = "📜BaBaAi Tools/QwenImageEdit文本编码器"

    def crop_and_scale(self, image, pad_info, enable_scaling=True, scale_method="lanczos", latent_upscale_factor=1.0):
        
        x = pad_info.get("x", 0)
        y = pad_info.get("y", 0)
        width_padding = pad_info.get("width", 0)
        height_padding = pad_info.get("height", 0)
        scale_by = pad_info.get("scale_by", 1.0)

        scaled_width_padding = round(width_padding * latent_upscale_factor)
        scaled_height_padding = round(height_padding * latent_upscale_factor)

        img = image.movedim(-1, 1)

        original_content_width = img.shape[3] - scaled_width_padding
        original_content_height = img.shape[2] - scaled_height_padding

        cropped_img = img[:, :, y:original_content_height, x:original_content_width]
        
        if enable_scaling:
            
            if scale_by == 0.0 or latent_upscale_factor == 0.0:
                print("[BaBaAi-CropWithPadInfo] 警告: scale_by 或 latent_upscale_factor 因子为 0。返回未缩放的裁切图像。")
                final_tensor = cropped_img
            else:
                final_scale_factor = scale_by / latent_upscale_factor
                _b, _c, current_h, current_w = cropped_img.shape
                target_w = int(round(current_w * final_scale_factor))
                target_h = int(round(current_h * final_scale_factor))

                if target_w < 1 or target_h < 1:
                     print(f"[BaBaAi-CropWithPadInfo] 警告: 目标尺寸小于 1 ({target_w}x{target_h})。已修正为 1x1。")
                     target_w = max(1, target_w)
                     target_h = max(1, target_h)
                
                rescaled_img_tensor = comfy.utils.common_upscale(cropped_img, target_w, target_h, scale_method, "disabled")
                final_tensor = rescaled_img_tensor
        else:
            final_tensor = cropped_img

        final_image = final_tensor.movedim(1, -1)
        
        return (final_image,)

NODE_CLASS_MAPPINGS = {
    "TextEncodeQwenImageEditPlus_BaBaAi": TextEncodeQwenImageEditPlus_BaBaAi,
    "TextEncodeQwenImageEditPlusAdvance_BaBaAi": TextEncodeQwenImageEditPlusAdvance_BaBaAi,
    "TextEncodeQwenImageEditPlusPro_BaBaAi": TextEncodeQwenImageEditPlusPro_BaBaAi,
    "CropWithPadInfo_BaBaAi": CropWithPadInfo_BaBaAi,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeQwenImageEditPlus_BaBaAi": "TextEncodeQwenImageEdit-Plus",
    "TextEncodeQwenImageEditPlusAdvance_BaBaAi": "TextEncodeQwenImageEdit-PlusAdvance",
    "TextEncodeQwenImageEditPlusPro_BaBaAi": "TextEncodeQwenImageEdit-PlusPro",
    "CropWithPadInfo_BaBaAi": "CropWithPadInfo-Plus",
}