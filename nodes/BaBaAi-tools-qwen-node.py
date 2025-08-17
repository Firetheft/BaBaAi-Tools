import os
import uuid
import folder_paths
import numpy as np

from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer
)
from pathlib import Path
from comfy_api.input import VideoInput
import torch
import re
import random
import time
from collections import Counter


# --- 自定义 TextStreamer 类 (从用户提供的代码中提取) ---
class CustomTextStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=True):
        super().__init__(tokenizer, skip_prompt=skip_prompt, skip_special_tokens=skip_special_tokens)
        self.generated_text = ""
        self.stop_flag = False
        self.init_time = time.time()  # Record initialization time
        self.end_time = None  # To store end time
        self.first_token_time = None  # To store first token generation time
        self.token_count = 0  # To track total tokens

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if self.first_token_time is None and text.strip():  # Set first token time on first non-empty text
            self.first_token_time = time.time()
        self.generated_text += text
        # Count tokens in the generated text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        self.token_count += len(tokens)
        
        if stream_end:
            self.end_time = time.time()  # Record end time when streaming ends
        if self.stop_flag:
            raise StopIteration

    def stop_generation(self):
        self.stop_flag = True
        self.end_time = time.time()  # Record end time when generation is stopped

    def get_metrics(self):
        """Returns initialization time, first token time, first token latency, end time, total time, total tokens, and tokens per second."""
        if self.end_time is None:
            self.end_time = time.time()  # Set end time if not already set
        total_time = self.end_time - self.init_time  # Total time from init to end
        tokens_per_second = self.token_count / total_time if total_time > 0 else 0
        first_token_latency = (self.first_token_time - self.init_time) if self.first_token_time is not None else None
        
        metrics = {
            "init_time": self.init_time,
            "first_token_time": self.first_token_time,
            "first_token_latency": first_token_latency,
            "end_time": self.end_time,
            "total_time": total_time,  # Total time in seconds
            "total_tokens": self.token_count,
            "tokens_per_second": tokens_per_second
        }
        
        metrics_str = f"Metrics:\n"
        for key, value in metrics.items():
            metrics_str += f"  {key}: {value}\n"

        return metrics_str


# 辅助函数 (从原 ComfyUI-Qwen2_5-VL/nodes.py 复制)
def temp_video(video: VideoInput, seed):
    unique_id = uuid.uuid4().hex
    video_path = (
        Path(folder_paths.temp_directory) / f"temp_video_{seed}_{unique_id}.mp4"
    )
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video.save_to(
        os.path.join(video_path),
        format="mp4",
        codec="h264",
    )

    uri = f"{video_path.as_posix()}"
    return uri


def temp_image(image, seed):
    unique_id = uuid.uuid4().hex
    image_path = (
        Path(folder_paths.temp_directory) / f"temp_image_{seed}_{unique_id}.png"
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )
    img.save(os.path.join(image_path))

    uri = f"file://{image_path.as_posix()}"
    return uri


def temp_batch_image(image, num_counts, seed):
    image_batch_path = Path(folder_paths.temp_directory) / "Multiple"
    image_batch_path.mkdir(parents=True, exist_ok=True)
    image_paths = []

    for Nth_count in range(num_counts):
        img = Image.fromarray(
            np.clip(255.0 * image[Nth_count].cpu().numpy().squeeze(), 0, 255).astype(
                np.uint8
            )
        )
        unique_id = uuid.uuid4().hex
        image_path = image_batch_path / f"temp_image_{seed}_{Nth_count}_{unique_id}.png"
        img.save(os.path.join(image_path))

        image_paths.append(f"file://{image_path.resolve().as_posix()}")

    return image_paths


model_directory = os.path.join(folder_paths.models_dir, "VLM")
os.makedirs(model_directory, exist_ok=True)


class Qwen2_5_VLModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "Qwen/Qwen2.5-VL-3B-Instruct",
                        "huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated",
                        "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
                        "Qwen/Qwen2.5-VL-7B-Instruct",
                        "huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated",
                        "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
                    ],
                    {"default": "Qwen/Qwen2.5-VL-3B-Instruct"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "8bit"},
                ),
                "attention": (
                    ["flash_attention_2", "sdpa", "eager"],
                    {"default": "sdpa"},
                ),
            },
        }

    RETURN_TYPES = ("QWEN2_5_VL_MODEL",)
    RETURN_NAMES = ("Qwen2_5_VL_model",)
    FUNCTION = "load_model"
    CATEGORY = "📜BaBaAi Tools"

    def load_model(self, model, quantization, attention):
        Qwen2_5_VL_model = {"model": "", "model_path": ""}
        model_name = model.rsplit("/", 1)[-1]
        model_path = os.path.join(model_directory, model_name)

        if not os.path.exists(model_path):
            print(f"Downloading Qwen2.5VL model to: {model_path}")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model, local_dir=model_path, local_dir_use_symlinks=False
            )

        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            quantization_config = None

        Qwen2_5_VL_model["model"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation=attention,
            quantization_config=quantization_config,
        )
        Qwen2_5_VL_model["model_path"] = model_path

        return (Qwen2_5_VL_model,)


class Qwen2_5_VL_Run_Advanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
            },
            "required": {
                "system_text": ("STRING", {"default": "", "multiline": True}),
                "preset_prompt": ([
                    "无指令",
                    "文本指令-提示词增强（Flux）",
                    "文本指令-提示词增强（Flux中文）",
                    "文本指令-提示词增强（标签）",
                    "文本指令-提示词增强（Wan）",
                    "文本指令-提示词增强（Wan中文）",
                    "文本指令-文本打标",
                    "文本指令-翻译成中文",
                    "文本指令-翻译成英文",
                    "文本指令-小说结构化",
                    "图片指令-图片描述",
                    "图片指令-图片描述（中文）",
                    "图片指令-图片打标",
                    "图片指令-首尾过度（Wan）",
                    "图片指令-图片描述（Wan）",
                    "图片指令-图片颜色",
                    "图片指令-图片HEX",
                    "图片指令-图片RGB",
                    "视频指令-视频描述",
                    "视频指令-默片补音",
                    "音频指令-音频描述"
                ], {"default": "无指令"}),
                "text": ("STRING", {"default": "", "multiline": True}),
                "Qwen2_5_VL_model": ("QWEN2_5_VL_MODEL",),
                "video_decode_method": (
                    ["torchvision", "decord", "torchcodec"],
                    {"default": "torchvision"},
                ),
                "max_new_tokens": ("INT", {"default": 128, "min": 1, "max": 1024}),
                "min_pixels": (
                    "INT",
                    {
                        "default": 256,
                        "min": 64,
                        "max": 1280,
                        "tooltip": "Define min_pixels and max_pixels: Images will be resized to maintain their aspect ratio within the range of min_pixels and max_pixels.",
                    },
                ),
                "max_pixels": (
                    "INT",
                    {
                        "default": 1280,
                        "min": 64,
                        "max": 2048,
                        "tooltip": "Define min_pixels and max_pixels: Images will be resized to maintain their aspect ratio within the range of min_pixels and max_pixels.",
                    },
                ),
                "total_pixels": (
                    "INT",
                    {
                        "default": 20480,
                        "min": 1,
                        "max": 24576,
                        "tooltip": "We recommend setting appropriate values for the min_pixels and max_pixels parameters based on available GPU memory and the specific application scenario to restrict the resolution of individual frames in the video. Alternatively, you can use the total_pixels parameter to limit the total number of tokens in the video (it is recommended to set this value below 24576 * 28 * 28 to avoid excessively long input sequences). For more details on parameter usage and processing logic, please refer to the fetch_video function in qwen_vl_utils/vision_process.py.",
                    },
                ),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "Qwen2_5_VL_Run_Advanced"
    CATEGORY = "📜BaBaAi Tools"

    def Qwen2_5_VL_Run_Advanced(
        self,
        system_text,
        preset_prompt,
        text,
        Qwen2_5_VL_model,
        video_decode_method,
        max_new_tokens,
        min_pixels,
        max_pixels,
        total_pixels,
        seed,
        image=None,
        video=None,
    ):
        preset_prompt_map = {
            "无指令": "",
            "文本指令-提示词增强（Flux）": "You are a prompt enhancer for the Flux image generation model. Given a short Chinese phrase describing any subject—person, animal, object, or scene—translate it into English and expand it into a single, richly detailed and visually evocative prompt. By default, generate descriptions in the style of realistic photography, including details such as subject appearance, pose, clothing or texture, environment, lighting, mood, and camera perspective. If the original phrase explicitly mentions an artistic style (e.g., flat illustration, oil painting, cyberpunk), adapt the description to match that style instead. Do not provide multiple options or explanations—just output one complete, vivid paragraph suitable for high-quality image synthesis.",
            "文本指令-提示词增强（Flux中文）": "You are a prompt enhancement assistant for the Flux image generation model. Users will enter a short Chinese phrase, possibly describing a person, animal, object, or scene. Please expand this phrase into a complete, detailed, and visually expressive Chinese prompt. By default, please describe the subject in a realistic photographic style, including details such as the subject's appearance, pose, clothing or material, environment, lighting, atmosphere, and camera angle. If the original phrase explicitly mentions an artistic style (such as flat illustration, oil painting, cyberpunk, etc.), please construct the prompt based on that style. Do not provide multiple options or explanations, and only output a complete Chinese prompt suitable for high-quality image generation.",
            "文本指令-提示词增强（标签）": "Expand the following input into tag-style phrases suitable for Stable Diffusion 1.5 or SDXL. The output must consist of short, descriptive English keywords only—no full sentences or natural language. Follow this structure and order: Character Features (e.g., age, gender, appearance, clothing), Environment (e.g., background, setting), Lighting (e.g., time of day, light type), View Angle (e.g., close-up, wide shot, top-down), and Image Quality (e.g., high quality, 8k, cinematic). If the original input is in Chinese, translate it into English first. Only generate one set of tags—do not provide multiple options or explanations.",
            "文本指令-提示词增强（Wan）": "You are a cinematic prompt composer for the wan2.2 video generation model. Based on the following simplified input text, expand it into a fully detailed video prompt. Your output must include descriptive elements for (1) subject appearance and identity, (2) scene and environment, (3) specific motion/action events, (4) cinematic composition and camera behavior, (5) lighting and color palette, (6) stylization and aesthetic tone. Construct a visually rich, dramatic and filmic scene suitable for high-quality video synthesis. Use natural language with vivid imagery and maintain temporal continuity. Do not over-describe static elements—favor dynamic and expressive language. Do not add any conversational text, explanations, or deviations",
            "文本指令-提示词增强（Wan中文）": "You are a cinematic prompt composer for the wan2.2 video generation model. Based on the following simplified input text, expand it into a fully detailed video Chinese prompt. Your output must include descriptive elements for (1) subject appearance and identity, (2) scene and environment, (3) specific motion/action events, (4) cinematic composition and camera behavior, (5) lighting and color palette, (6) stylization and aesthetic tone. Construct a visually rich, dramatic and filmic scene suitable for high-quality video synthesis. Use natural language with vivid imagery and maintain temporal continuity. Do not over-describe static elements—favor dynamic and expressive language. Do not add any conversational text, explanations, or deviations, and only output a complete Chinese prompt suitable for high-quality video generation.",
            "文本指令-文本打标": "Transform the text into tags, separated by commas, don’t use duplicate tags",
            "文本指令-翻译成中文": "Translate this text into Chinese like translation software, and the results only need to be displayed in Chinese",
            "文本指令-翻译成英文": "Translate this text into English like translation software, and the results only need to be displayed in English",
            # --- 新加入的指令 ---
            "文本指令-小说结构化": """你是一个专业的小说文本结构化处理器。请严格按以下规则处理输入文本：

1. 将文本拆分为`<Narrator>`叙述段落和`<CharacterX>`角色对话

2. 角色分配规则：
    - 同一角色始终使用相同Character编号（如王野始终是<Character1>）
    - 新角色首次出现时自动分配新编号
    - 角色识别优先级：角色名 > 代词(他/她) > 特征描述
    - 如特征模糊请从整个文本内容去分析如何分配编号

3. 文本分类规则：
    - 直接引语归入角色标签
    - 动作/环境/心理描写归入Narrator
    - 对话引导语（如"王野说道："）归入Narrator

4. 输出格式要求：
    - 每段独立一行，用指定标签包裹
    - 严格保持原文标点和换行
    - 不添加任何额外说明或注释

### 输出示例
输入文本：
'''
少女此时就站在院墙那边，她有一双杏眼，怯怯弱弱。
院门那边，有个嗓音说：“你这婢女卖不卖？”
宋集薪愣了愣，循着声音转头望去，是个眉眼含笑的锦衣少年，站在院外，一张全然陌生的面孔。
锦衣少年身边站着一位身材高大的老者，面容白皙，脸色和蔼，轻轻眯眼打量着两座毗邻院落的少年少女。
老者的视线在陈平安一扫而过，并无停滞，但是在宋集薪和婢女身上，多有停留，笑意渐渐浓郁。
宋集薪斜眼道：“卖！怎么不卖！”
那少年微笑道：“那你说个价。”
少女瞪大眼眸，满脸匪夷所思，像一头惊慌失措的年幼麋鹿。
宋集薪翻了个白眼，伸出一根手指，晃了晃，“白银一万两！”
锦衣少年脸色如常，点头道：“好。”
宋集薪见那少年不像是开玩笑的样子，连忙改口道：“是黄金万两！”
锦衣少年嘴角翘起，道：“逗你玩的。”
宋集薪脸色阴沉。
'''

正确输出：
<Narrator>少女此时就站在院墙那边，她有一双杏眼，怯怯弱弱。</Narrator>
<Narrator>院门那边，有个嗓音说：</Narrator>
<Character2>“你这婢女卖不卖？”</Character2>
<Narrator>宋集薪愣了愣，循着声音转头望去，是个眉眼含笑的锦衣少年，站在院外，一张全然陌生的面孔。</Narrator>
<Narrator>锦衣少年身边站着一位身材高大的老者，面容白皙，脸色和蔼，轻轻眯眼打量着两座毗邻院落的少年少女。</Narrator>
<Narrator>老者的视线在陈平安一扫而过，并无停滞，但是在宋集薪和婢女身上，多有停留，笑意渐渐浓郁。</Narrator>
<Narrator>宋集薪斜眼道：</Narrator>
<Character1>“卖！怎么不卖！”</Character1>
<Narrator>那少年微笑道：</Narrator>
<Character2>“那你说个价。”</Character2>
<Narrator>少女瞪大眼眸，满脸匪夷所思，像一头惊慌失措的年幼麋鹿。</Narrator>
<Narrator>宋集薪翻了个白眼，伸出一根手指，晃了晃，</Narrator>
<Character1>“白银一万两！”</Character1>
<Narrator>锦衣少年脸色如常，点头道：</Narrator>
<Character2>“好。”</Character2>
<Narrator>宋集薪见那少年不像是开玩笑的样子，连忙改口道：</Narrator>
<Character1>“是黄金万两！”</Character1>
<Narrator>锦衣少年嘴角翘起，道：</Narrator>
<Character2>“逗你玩的。”</Character2>
<Narrator>宋集薪脸色阴沉。</Narrator>""",
            # --------------------            
            "图片指令-图片描述": "Describe the image in detail and accurately",
            "图片指令-图片描述（中文）": "详细描叙这张图片的内容",
            "图片指令-图片打标": "Use tags to briefly and accurately describe this image, separated by commas, don’t use duplicate tags",
            "图片指令-首尾过度（Wan）": "You are a cinematic prompt composer for the wan2.2 video generation model. You are given a composite image containing two frames: the right side is the starting frame of the video, and the left side is the ending frame. Your task is to analyze both frames and imagine a smooth, emotionally coherent transition between them. Generate a single, detailed video prompt that describes the full temporal arc—from the initial pose to the final pose—by constructing plausible intermediate actions, gestures, expressions, and environmental changes. Include subject movement, camera behavior, framing, lighting evolution, and pacing. The transition must feel natural and continuous, without abrupt jumps. Do not explain your reasoning—just output the final prompt.",
            "图片指令-图片描述（Wan）": "You are a cinematic prompt composer for the wan2.2 video generation model. Based on the uploaded image, analyze its visual content and extrapolate a fully detailed video prompt. Your output must incorporate accurate and creative inferences of (1) subject appearance and identity, (2) setting and environmental design, (3) motion or action, (4) cinematic composition and camera behavior, (5) lighting and mood, and (6) stylization and aesthetic tone. Frame the output as a dynamic scene suitable for temporal video generation. Use vivid imagery, expressive verbs, and avoid static descriptions. Highlight subtle emotional nuance and visual storytelling potential. Do not over-describe static elements—favor dynamic and expressive language. Do not add any conversational text, explanations, or deviations",
            "图片指令-图片颜色": "Analyze the image and extract the plain_english_colors of the 5 main colors, separated by commas",
            "图片指令-图片HEX": "Analyze the image and extract the HEX values of the 5 main colors, separated by commas",
            "图片指令-图片RGB": "Analyze the image and extract the RGB values of the 5 main colors, separated by commas and parentheses",
            "视频指令-视频描述": "Describe the video in detail and accurately",
            "视频指令-默片补音": "You are a voice synthesis prompt composer for the MMAudio model. Your task is to generate realistic and expressive voice-based audio for a silent video. Analyze the visual content frame by frame, and for each segment, determine the appropriate sound—whether it's speech, ambient noise, mechanical sounds, emotional vocalizations, or symbolic utterances. Your output must reflect the timing, pacing, and emotional tone of the video, matching visual actions and transitions precisely. Avoid vague or generic words; instead, write full, meaningful voice or sound content that would naturally accompany the visuals. Do not add any conversational text, explanations, deviations, or numbering.",
            "音频指令-音频描述": "Describe the audio in detail and accurately"
        }

        min_pixels = min_pixels * 28 * 28
        max_pixels = max_pixels * 28 * 28
        total_pixels = total_pixels * 28 * 28

        processor = AutoProcessor.from_pretrained(Qwen2_5_VL_model["model_path"])

        # Combine preset prompt and user text
        if preset_prompt != "无指令":
            full_text = f"{preset_prompt_map.get(preset_prompt, '')}\n{text}"
        else:
            full_text = text

        content = []
        if image is not None:
            num_counts = image.shape[0]
            if num_counts == 1:
                uri = temp_image(image, seed)
                content.append(
                    {
                        "type": "image",
                        "image": uri,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    }
                )
            elif num_counts > 1:
                image_paths = temp_batch_image(image, num_counts, seed)
                for path in image_paths:
                    content.append(
                        {
                            "type": "image",
                            "image": path,
                            "min_pixels": min_pixels,
                            "max_pixels": max_pixels,
                        }
                    )

        if video is not None:
            uri = temp_video(video, seed)
            content.append(
                {
                    "type": "video",
                    "video": uri,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "total_pixels": total_pixels,
                }
            )

        if full_text:
            content.append({"type": "text", "text": full_text})

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": content},
        ]
        modeltext = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        os.environ["FORCE_QWENVL_VIDEO_READER"] = video_decode_method
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        inputs = processor(
            text=[modeltext],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(Qwen2_5_VL_model["model"].device)
        generated_ids = Qwen2_5_VL_model["model"].generate(
            **inputs, max_new_tokens=max_new_tokens
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return (str(output_text),)
    
class Qwen3ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (
                    [
                        "Qwen/Qwen3-4B-Instruct-2507-FP8",
                        "huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated"
                    ],
                    {"default": "Qwen/Qwen3-4B-Instruct-2507-FP8"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "4bit"},
                ),
                "attention": (
                    ["flash_attention_2", "sdpa", "eager"],
                    {"default": "sdpa"},
                ),
            },
        }

    RETURN_TYPES = ("QWEN3_MODEL",)
    RETURN_NAMES = ("Qwen3_model",)
    FUNCTION = "load_model"
    CATEGORY = "📜BaBaAi Tools"

    def load_model(self, model_name, quantization, attention):
        model_id = model_name.rsplit("/", 1)[-1]
        model_path = os.path.join(model_directory, model_id)

        if not os.path.exists(model_path):
            print(f"Downloading {model_name} model to: {model_path}")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_name, local_dir=model_path, local_dir_use_symlinks=False
            )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="balanced",
                trust_remote_code=True,
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation=attention,
            )
        elif quantization == "8bit":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="balanced",
                trust_remote_code=True,
                quantization_config=quant_config,
                low_cpu_mem_usage=True,
                attn_implementation=attention,
            )
        else: # quantization == "none"
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="balanced",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation=attention,
            )
            
        return ({"model": model, "tokenizer": tokenizer},)
    
class Qwen3_Run_Advanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "Qwen3_model": ("QWEN3_MODEL",),
                "preset_prompt": ([
                    "无指令",
                    "文本指令-提示词增强（Flux）",
                    "文本指令-提示词增强（Flux中文）",
                    "文本指令-提示词增强（标签）",
                    "文本指令-提示词增强（Wan）",
                    "文本指令-提示词增强（Wan中文）",
                    "文本指令-文本打标",
                    "文本指令-翻译成中文",
                    "文本指令-翻译成英文",
                    "文本指令-小说结构化",
                ], {"default": "无指令"}),
                "max_new_tokens": ("INT", {"default": 16384, "min": 1, "max": 16384}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "设置为0时，每次运行生成随机种子。"}),
            },
            "optional": {
                "bypass_model": ("BOOLEAN", {"default": False, "label_on": "Bypass Model", "label_off": "Use Model"})
            }
        }

    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("text", "metrics")
    FUNCTION = "run_model"
    CATEGORY = "📜BaBaAi Tools"

    def run_model(self, text, Qwen3_model, preset_prompt, max_new_tokens, temperature, top_p, top_k, seed, bypass_model=False):
        
        if bypass_model:
            return (text,"")

        preset_prompt_map = {
            "无指令": "",
            "文本指令-提示词增强（Flux）": "You are a prompt enhancer for the Flux image generation model. Given a short Chinese phrase describing any subject—person, animal, object, or scene—translate it into English and expand it into a single, richly detailed and visually evocative prompt. By default, generate descriptions in the style of realistic photography, including details such as subject appearance, pose, clothing or texture, environment, lighting, mood, and camera perspective. If the original phrase explicitly mentions an artistic style (e.g., flat illustration, oil painting, cyberpunk), adapt the description to match that style instead. Do not provide multiple options or explanations—just output one complete, vivid paragraph suitable for high-quality image synthesis.",
            "文本指令-提示词增强（Flux中文）": "You are a prompt enhancement assistant for the Flux image generation model. Users will enter a short Chinese phrase, possibly describing a person, animal, object, or scene. Please expand this phrase into a complete, detailed, and visually expressive Chinese prompt. By default, please describe the subject in a realistic photographic style, including details such as the subject's appearance, pose, clothing or material, environment, lighting, atmosphere, and camera angle. If the original phrase explicitly mentions an artistic style (such as flat illustration, oil painting, cyberpunk, etc.), please construct the prompt based on that style. Do not provide multiple options or explanations, and only output a complete Chinese prompt suitable for high-quality image generation.",
            "文本指令-提示词增强（标签）": "Expand the following input into tag-style phrases suitable for Stable Diffusion 1.5 or SDXL. The output must consist of short, descriptive English keywords only—no full sentences or natural language. Follow this structure and order: Character Features (e.g., age, gender, appearance, clothing), Environment (e.g., background, setting), Lighting (e.g., time of day, light type), View Angle (e.g., close-up, wide shot, top-down), and Image Quality (e.g., high quality, 8k, cinematic). If the original input is in Chinese, translate it into English first. Only generate one set of tags—do not provide multiple options or explanations.",
            "文本指令-提示词增强（Wan）": "You are a cinematic prompt composer for the wan2.2 video generation model. Based on the following simplified input text, expand it into a fully detailed video prompt. Your output must include descriptive elements for (1) subject appearance and identity, (2) scene and environment, (3) specific motion/action events, (4) cinematic composition and camera behavior, (5) lighting and color palette, (6) stylization and aesthetic tone. Construct a visually rich, dramatic and filmic scene suitable for high-quality video synthesis. Use natural language with vivid imagery and maintain temporal continuity. Do not over-describe static elements—favor dynamic and expressive language. Do not add any conversational text, explanations, or deviations.",
            "文本指令-提示词增强（Wan中文）": "You are a cinematic prompt composer for the wan2.2 video generation model. Based on the following simplified input text, expand it into a fully detailed video Chinese prompt. Your output must include descriptive elements for (1) subject appearance and identity, (2) scene and environment, (3) specific motion/action events, (4) cinematic composition and camera behavior, (5) lighting and color palette, (6) stylization and aesthetic tone. Construct a visually rich, dramatic and filmic scene suitable for high-quality video synthesis. Use natural language with vivid imagery and maintain temporal continuity. Do not over-describe static elements—favor dynamic and expressive language. Do not add any conversational text, explanations, or deviations, and only output a complete Chinese prompt suitable for high-quality video generation.",
            "文本指令-文本打标": "Transform the text into tags, separated by commas, don’t use duplicate tags",
            "文本指令-翻译成中文": "Translate this text into Chinese like translation software, and the results only need to be displayed in Chinese",
            "文本指令-翻译成英文": "Translate this text into English like translation software, and the results only need to be displayed in English",
            # --- 新加入的指令 ---
            "文本指令-小说结构化": """你是一个专业的小说文本结构化处理器。请严格按以下规则处理输入文本：

1. 将文本拆分为`<Narrator>`叙述段落和`<CharacterX>`角色对话

2. 角色分配规则：
    - 同一角色始终使用相同Character编号（如王野始终是<Character1>）
    - 新角色首次出现时自动分配新编号
    - 角色识别优先级：角色名 > 代词(他/她) > 特征描述
    - 如特征模糊请从整个文本内容去分析如何分配编号

3. 文本分类规则：
    - 直接引语归入角色标签
    - 动作/环境/心理描写归入Narrator
    - 对话引导语（如"王野说道："）归入Narrator

4. 输出格式要求：
    - 每段独立一行，用指定标签包裹
    - 严格保持原文标点和换行
    - 不添加任何额外说明或注释

### 输出示例
输入文本：
'''
少女此时就站在院墙那边，她有一双杏眼，怯怯弱弱。
院门那边，有个嗓音说：“你这婢女卖不卖？”
宋集薪愣了愣，循着声音转头望去，是个眉眼含笑的锦衣少年，站在院外，一张全然陌生的面孔。
锦衣少年身边站着一位身材高大的老者，面容白皙，脸色和蔼，轻轻眯眼打量着两座毗邻院落的少年少女。
老者的视线在陈平安一扫而过，并无停滞，但是在宋集薪和婢女身上，多有停留，笑意渐渐浓郁。
宋集薪斜眼道：“卖！怎么不卖！”
那少年微笑道：“那你说个价。”
少女瞪大眼眸，满脸匪夷所思，像一头惊慌失措的年幼麋鹿。
宋集薪翻了个白眼，伸出一根手指，晃了晃，“白银一万两！”
锦衣少年脸色如常，点头道：“好。”
宋集薪见那少年不像是开玩笑的样子，连忙改口道：“是黄金万两！”
锦衣少年嘴角翘起，道：“逗你玩的。”
宋集薪脸色阴沉。
'''

正确输出：
<Narrator>少女此时就站在院墙那边，她有一双杏眼，怯怯弱弱。</Narrator>
<Narrator>院门那边，有个嗓音说：</Narrator>
<Character2>“你这婢女卖不卖？”</Character2>
<Narrator>宋集薪愣了愣，循着声音转头望去，是个眉眼含笑的锦衣少年，站在院外，一张全然陌生的面孔。</Narrator>
<Narrator>锦衣少年身边站着一位身材高大的老者，面容白皙，脸色和蔼，轻轻眯眼打量着两座毗邻院落的少年少女。</Narrator>
<Narrator>老者的视线在陈平安一扫而过，并无停滞，但是在宋集薪和婢女身上，多有停留，笑意渐渐浓郁。</Narrator>
<Narrator>宋集薪斜眼道：</Narrator>
<Character1>“卖！怎么不卖！”</Character1>
<Narrator>那少年微笑道：</Narrator>
<Character2>“那你说个价。”</Character2>
<Narrator>少女瞪大眼眸，满脸匪夷所思，像一头惊慌失措的年幼麋鹿。</Narrator>
<Narrator>宋集薪翻了个白眼，伸出一根手指，晃了晃，</Narrator>
<Character1>“白银一万两！”</Character1>
<Narrator>锦衣少年脸色如常，点头道：</Narrator>
<Character2>“好。”</Character2>
<Narrator>宋集薪见那少年不像是开玩笑的样子，连忙改口道：</Narrator>
<Character1>“是黄金万两！”</Character1>
<Narrator>锦衣少年嘴角翘起，道：</Narrator>
<Character2>“逗你玩的。”</Character2>
<Narrator>宋集薪脸色阴沉。</Narrator>""",
            # --------------------
        }

        # Combine preset prompt and user text
        if preset_prompt != "无指令":
            full_text = f"{preset_prompt_map.get(preset_prompt, '')}\n{text}"
        else:
            full_text = text

        tokenizer = Qwen3_model["tokenizer"]
        model = Qwen3_model["model"]

        messages = [{"role": "user", "content": full_text}]
        model_inputs = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([model_inputs], return_tensors="pt").to(model.device)

        # Handle seed
        do_sample = temperature > 0.0
        if seed == 0:
            seed = random.randint(0, 0xFFFFFFFFFFFFFFFF)
        
        # 实例化自定义的 Streamer 来捕获输出和指标
        streamer = CustomTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generate_kwargs = {
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 2,
            "streamer": streamer,
        }
        
        if do_sample:
            generate_kwargs.update({
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            })
            
        print("Response: ", end="", flush=True)

        try:
            generated_ids = model.generate(
                **model_inputs,
                **generate_kwargs
            )
            output_text = streamer.generated_text
            metrics_text = streamer.get_metrics()
            
            # 清理生成的文本中可能包含的 <|im_end|> 标记
            output_text = re.sub(r'<\|im_end\|>', '', output_text)
            print("\n")
            print(metrics_text, flush=True)

        except Exception as e:
            output_text = f"Error during generation: {e}"
            metrics_text = ""
            print(f"Error: {e}")

        return (output_text, metrics_text)

NODE_CLASS_MAPPINGS = {
    "Qwen2_5_VLModelLoader": Qwen2_5_VLModelLoader,
    "Qwen2_5_VL_Run_Advanced": Qwen2_5_VL_Run_Advanced,
    "Qwen3ModelLoader": Qwen3ModelLoader,
    "Qwen3_Run_Advanced": Qwen3_Run_Advanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen2_5_VLModelLoader": "Qwen2.5-VL模型加载器",
    "Qwen2_5_VL_Run_Advanced": "Qwen2.5-VL高级运行器",
    "Qwen3ModelLoader": "Qwen3模型加载器",
    "Qwen3_Run_Advanced": "Qwen3高级运行器",
}