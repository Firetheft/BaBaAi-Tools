import os
import json
import base64
import google.generativeai as genai
from io import BytesIO
from PIL import Image
import torch
import torchaudio
import numpy as np
import re

p = os.path.dirname(os.path.realpath(__file__))

def get_config():
    try:
        config_path = os.path.join(p, 'config.json')
        with open(config_path, 'r') as f:  
            config = json.load(f)
        return config
    except:
        return {}

def save_config(config):
    config_path = os.path.join(p, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

class ChatHistory:
    def __init__(self):
        self.messages = []
    
    def add_message(self, role, content):
        if isinstance(content, list):
            content = " ".join(str(item) for item in content if isinstance(item, str))
        self.messages.append({"role": role, "content": content})
    
    def get_formatted_history(self):
        formatted = "\n=== Chat History ===\n"
        for msg in self.messages:
            formatted += f"{msg['role'].upper()}: {msg['content']}\n"
        formatted += "=== End History ===\n"
        return formatted
    
    def get_messages_for_api(self):
        api_messages = []
        for msg in self.messages:
            if isinstance(msg["content"], str):
                api_messages.append({
                    "role": msg["role"],
                    "parts": [{"text": msg["content"]}]
                })
        return api_messages
    
    def clear(self):
        self.messages = []

class GeminiFlash:
    def __init__(self, api_key=None):
        env_key = os.environ.get("GEMINI_API_KEY")

        placeholders = {"token_here", "place_token_here", "your_api_key",
                        "api_key_here", "enter_your_key", "<api_key>"}

        if env_key and env_key.lower().strip() not in placeholders:
            self.api_key = env_key
        else:
            self.api_key = api_key

            if self.api_key is None:
                config = get_config()
                self.api_key = config.get("GEMINI_API_KEY")

        self.chat_history = ChatHistory()
        if self.api_key is not None:
            self.configure_genai()

    def configure_genai(self):
        genai.configure(api_key=self.api_key, transport='rest')

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "Analyze the situation in details.", "multiline": True}),
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
                    "图片指令-图片描述（图文）",
                    "图片指令-下一个镜头（标准）",
                    "图片指令-下一个镜头（强烈）",
                    "图片指令-下一个镜头（标准V2）",
                    "图片指令-下一个镜头（强烈V2）",
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
                "kontext_prompt": ([
                    "无指令",
                    "图片指令-图片编辑-传送",
                    "图片指令-图片编辑-移动相机",
                    "图片指令-图片编辑-重打光",
                    "图片指令-图片编辑-重构图",
                    "图片指令-图片编辑-产品摄影",
                    "图片指令-图片编辑-产品外观",
                    "图片指令-图片编辑-缩放",
                    "图片指令-图片编辑-上色",
                    "图片指令-图片编辑-LOGO设计",
                    "图片指令-图片编辑-尾帧图像",
                    "图片指令-图片编辑-电影海报",
                    "图片指令-图片编辑-卡通化",
                    "图片指令-图片编辑-移除文字",
                    "图片指令-图片编辑-发型更换",
                    "图片指令-图片编辑-服装更换",
                    "图片指令-图片编辑-写真达人",
                    "图片指令-图片编辑-健身达人",
                    "图片指令-图片编辑-移除家具",
                    "图片指令-图片编辑-室内设计",
                    "图片指令-图片编辑-建筑外观",
                    "图片指令-图片编辑-清理杂物",
                    "图片指令-图片编辑-艺术风格",
                    "图片指令-图片编辑-材质转换",
                    "图片指令-图片编辑-情绪变化",
                    "图片指令-图片编辑-年龄变化",
                    "图片指令-图片编辑-季节变化",
                    "图片指令-图片编辑-合成融合",
                    "图片指令-图片编辑-调色板",
                ], {"default": "无指令"}),
                "kontext_prompt_no_api": ([
                    "无指令",
                    "角色编辑 - 表情",
                    "角色编辑 - 动作",
                    "角色编辑 - 皮肤",
                ], {"default": "无指令"}),
                "input_type": (["text", "image", "video", "audio"], {"default": "text"}),
                "model_version": (["no-api", "gemini-2.5-flash", "gemini-2.0-flash-thinking-exp-1219", "gemini-2.5-flash-image-preview"], {"default": "gemini-2.5-flash"}),
                "operation_mode": (["analysis", "generate_images"], {"default": "analysis"}),
                "chat_mode": ("BOOLEAN", {"default": False}),
                "clear_history": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "Additional_Context": ("STRING", {"default": "", "multiline": True}),
                "images": ("IMAGE", {"forceInput": False, "list": True}),
                "video": ("IMAGE", ),
                "audio": ("AUDIO", ),
                "api_key": ("STRING", {"default": ""}),
                "max_output_tokens": ("INT", {"default": 8192, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1}),
                "structured_output": ("BOOLEAN", {"default": False}),
                "max_images": ("INT", {"default": 6, "min": 1, "max": 16, "step": 1}),
                "batch_count": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("generated_content", "generated_images")
    FUNCTION = "generate_content"
    CATEGORY = "📜BaBaAi Tools"

    def tensor_to_image(self, tensor):
        tensor = tensor.cpu()
        if len(tensor.shape) == 4:
            if tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            else:
                tensor = tensor[0]
                
        image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
        image = Image.fromarray(image_np, mode='RGB')
        return image

    def resize_image(self, image, max_size):
        width, height = image.size
        if width > height:
            if width > max_size:
                height = int(max_size * height / width)
                width = max_size
        else:
            if height > max_size:
                width = int(max_size * width / height)
                height = max_size
        return image.resize((width, height), Image.LANCZOS)

    def sample_video_frames(self, video_tensor, num_samples=6):
        """Sample frames evenly from video tensor"""
        if len(video_tensor.shape) != 4:
            return None

        total_frames = video_tensor.shape[0]
        if total_frames <= num_samples:
            indices = range(total_frames)
        else:
            indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

        frames = []
        for idx in indices:
            frame = self.tensor_to_image(video_tensor[idx])
            frame = self.resize_image(frame, 512)
            frames.append(frame)
        return frames

    def prepare_content(self, prompt, input_type, Additional_Context=None, images=None, video=None, audio=None, max_images=6):
        if input_type == "text":
            text_content = prompt if not Additional_Context else f"{prompt}\n{Additional_Context}"
            return [{"text": text_content}]
                
        elif input_type == "image":
            all_images = []
            
            if images is not None:
                if isinstance(images, torch.Tensor):
                    if len(images.shape) == 4:
                        num_images = min(images.shape[0], max_images)
                        
                        for i in range(num_images):
                            pil_image = self.tensor_to_image(images[i])
                            pil_image = self.resize_image(pil_image, 1024)
                            all_images.append(pil_image)
                    else:
                        pil_image = self.tensor_to_image(images)
                        pil_image = self.resize_image(pil_image, 1024)
                        all_images.append(pil_image)

                elif isinstance(images, list):
                    for img_tensor in images[:max_images]:
                        pil_image = self.tensor_to_image(img_tensor)
                        pil_image = self.resize_image(pil_image, 1024)
                        all_images.append(pil_image)

            if all_images:
                if len(all_images) > 1:
                    modified_prompt = f"Analyze these {len(all_images)} images. {prompt} Please describe each image separately."
                else:
                    modified_prompt = prompt
                    
                parts = [{"text": modified_prompt}]
                
                for idx, img in enumerate(all_images):
                    img_byte_arr = BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64.b64encode(img_byte_arr).decode('utf-8')
                        }
                    })
                
                return [{"parts": parts}]
            else:
                raise ValueError("No valid images provided")
                
        elif input_type == "video" and video is not None:
            frames = self.sample_video_frames(video)
            if frames:
                parts = [{"text": f"Analyzing video frames. {prompt}"}]
                for frame in frames:
                    img_byte_arr = BytesIO()
                    frame.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64.b64encode(img_byte_arr).decode('utf-8')
                        }
                    })
                return [{"parts": parts}]
            else:
                raise ValueError("Invalid video format")
                    
        elif input_type == "audio" and audio is not None:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)
            elif waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            
            buffer = BytesIO()
            torchaudio.save(buffer, waveform, 16000, format="WAV")
            audio_bytes = buffer.getvalue()
            
            return [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "audio/wav",
                            "data": base64.b64encode(audio_bytes).decode('utf-8')
                        }
                    }
                ]
            }]
        else:
            raise ValueError(f"Invalid or missing input for {input_type}")

    def create_placeholder_image(self):
        """Create a placeholder image tensor when generation fails"""
        img = Image.new('RGB', (512, 512), color=(73, 109, 137))
        image_array = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(image_array).unsqueeze(0)

    def generate_images(self, prompt, model_version, images=None, batch_count=1, temperature=0.4, seed=0, max_images=6):
        """Generate images using Gemini models with image generation capabilities"""
        try:
            is_image_generation_model = "image-generation" in model_version

            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=self.api_key)

            if is_image_generation_model:
                generation_config = types.GenerateContentConfig(
                    temperature=temperature,
                    response_modalities=['Text', 'Image']
                )
            else:
                generation_config = types.GenerateContentConfig(
                    temperature=temperature
                )

            content_parts = []
            if images is not None:
                all_images = []
                if isinstance(images, torch.Tensor):
                    if len(images.shape) == 4:
                        num_images = min(images.shape[0], max_images)
                        for i in range(num_images):
                            pil_image = self.tensor_to_image(images[i])
                            pil_image = self.resize_image(pil_image, 1024)
                            all_images.append(pil_image)
                    else:
                        pil_image = self.tensor_to_image(images)
                        pil_image = self.resize_image(pil_image, 1024)
                        all_images.append(pil_image)
                elif isinstance(images, list):
                    for img_tensor in images[:max_images]:
                        pil_image = self.tensor_to_image(img_tensor)
                        pil_image = self.resize_image(pil_image, 1024)
                        all_images.append(pil_image)
                
                if all_images:
                    if is_image_generation_model:
                        content_text = f"Generate a new image in the style of these reference images: {prompt}"
                    else:
                        content_text = f"Generate an image of: {prompt}"
                    
                    parts = [{"text": content_text}]
                    
                    for img in all_images:
                        img_byte_arr = BytesIO()
                        img.save(img_byte_arr, format='PNG')
                        img_bytes = img_byte_arr.getvalue()
                        
                        parts.append({
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": base64.b64encode(img_bytes).decode('utf-8')
                            }
                        })
                    
                    content_parts = [{"parts": parts}]
            else:
                if is_image_generation_model:
                    content_text = f"Generate a detailed, high-quality image of: {prompt}"
                else:
                    content_text = f"Generate an image of: {prompt}"
                
                content_parts = [{"parts": [{"text": content_text}]}]

            all_generated_images = []
            status_text = ""

            for i in range(batch_count):
                try:
                    if seed != 0:
                        current_seed = seed + i

                    response = client.models.generate_content(
                        model=model_version,
                        contents=content_parts,
                        config=generation_config
                    )

                    batch_images = []

                    response_text = ""
                    
                    if hasattr(response, 'candidates') and response.candidates:
                        for candidate in response.candidates:
                            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                for part in candidate.content.parts:
                                    if hasattr(part, 'text') and part.text:
                                        response_text += part.text + "\n"
                                    
                                    if hasattr(part, 'inline_data') and part.inline_data:
                                        try:
                                            image_binary = part.inline_data.data
                                            batch_images.append(image_binary)
                                        except Exception as img_error:
                                            print(f"Error extracting image from response: {str(img_error)}")
                    
                    if batch_images:
                        all_generated_images.extend(batch_images)
                        status_text += f"Batch {i+1}: Generated {len(batch_images)} images\n"
                    else:
                        status_text += f"Batch {i+1}: No images found in response. Text response: {response_text[:100]}...\n"
                
                except Exception as batch_error:
                    status_text += f"Batch {i+1} error: {str(batch_error)}\n"
            
            if all_generated_images:
                tensors = []
                for img_binary in all_generated_images:
                    try:
                        image = Image.open(BytesIO(img_binary))
                        
                        if image.mode != "RGB":
                            image = image.convert("RGB")

                        img_np = np.array(image).astype(np.float32) / 255.0

                        img_tensor = torch.from_numpy(img_np)[None,]
                        tensors.append(img_tensor)
                    except Exception as e:
                        print(f"Error processing image: {e}")
                
                if tensors:
                    image_tensors = torch.cat(tensors, dim=0)
                    
                    result_text = f"Successfully generated {len(tensors)} images using {model_version}.\n"
                    result_text += f"Prompt: {prompt}\n"
                    result_text += f"Details: {status_text}"
                    
                    return result_text, image_tensors
            
            return f"No images were generated with {model_version}. Details:\n{status_text}", self.create_placeholder_image()
            
        except Exception as e:
            error_msg = f"Error in image generation: {str(e)}"
            print(error_msg)
            return error_msg, self.create_placeholder_image()

    def generate_content(self, prompt, preset_prompt, kontext_prompt, kontext_prompt_no_api, input_type, model_version="gemini-2.5-flash", 
                        operation_mode="analysis", chat_mode=False, clear_history=False,
                        Additional_Context=None, images=None, video=None, audio=None, 
                        api_key="", max_images=6, batch_count=1, seed=0,
                        max_output_tokens=8192, temperature=0.4, structured_output=False):
        """Generate content using Gemini model with various input types."""

        instruction_base_single = "You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.\n\nThe brief:\n\n"
        common_suffix_single = "\n\nYour response must be a single, complete, and concise instruction ready for the image editing AI. Do not add any conversational text, explanations, deviations, or numbering."
        local_edit_suffix = "Finally, be sure to add the following after the prompt: This transformation must only affect the specified regio, Do not modify facial features, body structure, background, pose, lighting, or any other part of the image, Changes must be localized and preserve global consistency."

        preset_prompt_map = {
            "无指令": "",
            "文本指令-提示词增强（Flux）": "You are a prompt enhancer for the Flux image generation model. Given a short Chinese phrase describing any subject—person, animal, object, or scene—translate it into English and expand it into a single, richly detailed and visually evocative prompt. By default, generate descriptions in the style of realistic photography, including details such as subject appearance, pose, clothing or texture, environment, lighting, mood, and camera perspective. If the original phrase explicitly mentions an artistic style (e.g., flat illustration, oil painting, cyberpunk), adapt the description to match that style instead. Do not provide multiple options or explanations—just output one complete, vivid paragraph suitable for high-quality image synthesis.",
            "文本指令-提示词增强（Flux中文）": "You are a prompt enhancement assistant for the Flux image generation model. Users will enter a short Chinese phrase, possibly describing a person, animal, object, or scene. Please expand this phrase into a complete, detailed, and visually expressive Chinese prompt. By default, please describe the subject in a realistic photographic style, including details such as the subject's appearance, pose, clothing or material, environment, lighting, atmosphere, and camera angle. If the original phrase explicitly mentions an artistic style (such as flat illustration, oil painting, cyberpunk, etc.), please construct the prompt based on that style. Do not provide multiple options or explanations, and only output a complete Chinese prompt suitable for high-quality image generation.",
            "文本指令-提示词增强（标签）": "Expand the following prompt into tags for a Stable Diffusion prompt, following the structure and order, which is: Character Features, Environment, Lighting, View Angle, and Image Quality. Generate a single, creative description, do not provide multiple options, just write a single descriptive paragraph, if the original text is in Chinese, translate it into English first",
            "文本指令-提示词增强（Wan）": "You are a cinematic prompt composer for the wan2.2 video generation model. Based on the following simplified input text, expand it into a fully detailed video prompt. Your output must include descriptive elements for (1) subject appearance and identity, (2) scene and environment, (3) specific motion/action events, (4) cinematic composition and camera behavior, (5) lighting and color palette, (6) stylization and aesthetic tone. Construct a visually rich, dramatic and filmic scene suitable for high-quality video synthesis. Use natural language with vivid imagery and maintain temporal continuity. Do not over-describe static elements—favor dynamic and expressive language. Do not add any conversational text, explanations, or deviations",
            "文本指令-提示词增强（Wan中文）": "You are a cinematic prompt composer for the wan2.2 video generation model. Based on the following simplified input text, expand it into a fully detailed video Chinese prompt. Your output must include descriptive elements for (1) subject appearance and identity, (2) scene and environment, (3) specific motion/action events, (4) cinematic composition and camera behavior, (5) lighting and color palette, (6) stylization and aesthetic tone. Construct a visually rich, dramatic and filmic scene suitable for high-quality video synthesis. Use natural language with vivid imagery and maintain temporal continuity. Do not over-describe static elements—favor dynamic and expressive language. Do not add any conversational text, explanations, or deviations, and only output a complete Chinese prompt suitable for high-quality video generation.",
            "文本指令-文本打标": "Transform the text into tags, separated by commas, don’t use duplicate tags",
            "文本指令-翻译成中文": "Translate this text into Chinese like translation software, and the results only need to be displayed in Chinese",
            "文本指令-翻译成英文": "Translate this text into English like translation software, and the results only need to be displayed in English",
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
            "图片指令-图片描述": "Provide a vivid and insightful description of this image. Beyond just the visual content, capture its artistic style, any clear cultural references (like movies, cartoons, or IPs), and the overall mood or emotion. Please give only the direct description, free of labels, numbers, bullet points, or technical jargon",
            "图片指令-图片描述（中文）": "请用一段生动且富有洞察力的文字描述这幅图像。除了画面内容本身，还要捕捉到它的艺术风格、任何明显的文化引用（如电影、卡通、IP等），以及整体氛围或情感。请直接给出中文描述，不要包含任何标签、编号、项目符号或者技术性词语。",
            "图片指令-图片描述（图文）": "Provide a comprehensive and insightful analysis of this poster. Begin by capturing its overall artistic style, color scheme, composition, and visual mood. Crucially, read and seamlessly integrate the main text from the poster—such as titles, taglines, key information, or names—into your description. Explain how this textual information works together with the visual elements to convey a central theme or message. Present everything as a single, flowing paragraph, without using any labels, lists, or structural terms.",
            "图片指令-下一个镜头（标准）": """你是一个专业的电影导演助理和 AI 图像生成提示词专家。你的任务是为 next-scene-qwen-image-lora-2509 LoRA 模型生成高质量的提示词。
你的目标是生成一个“标准的下一个镜头”。这个镜头必须与前一帧有清晰、明确、合乎逻辑的进展。它不追求那种“戏剧性突变”，也避免那种“微小调整”。它模拟的是电影剪辑中一个标准的、有意义的镜头推进。

生成规则：

1，强制前缀： 每一个提示词都必须以英文短语 Next Scene: 开头（冒号后跟一个空格）。

2，核心内容（标准进展）： 提示词的主体必须清晰地描述一个**“单一且明确”**的变化。你必须从以下类别中选择一个作为主要驱动力：
    A. 清晰的摄像机运动 (Clear Camera Movement)： 描述一个完整的、有目的的摄像机运动。（避免使用“轻微/slightly”或“快速/rapidly”）
        关键词示例： The camera pushes in from a medium shot to a close-up on the character's hands... (镜头从中景推进到角色双手的特写), The camera pulls back to reveal the character is not alone... (镜头拉回，揭示角色不是一个人), The camera tracks alongside the character as they walk... (镜头跟随角色行走), The camera pans left to follow the car driving away... (镜头向左平移，跟随汽车驶离)。
    B. 明确的剪辑切换 (Clear Cut / Shot Change)： 明确描述一个新镜头的景别或角度，暗示这是一个“剪辑点”。
        关键词示例： The shot cuts to a close-up of the ringing phone... (镜头切到正在响铃的电话的特写), The shot is now a wide-angle establishing shot of the building... (现在的镜头是建筑物的广角定场镜头), The scene changes to an over-the-shoulder shot from behind the protagonist... (场景切换到主角背后的过肩镜头), A low-angle shot now makes the villain look intimidating... (一个低角度镜头现在让反派看起来很吓人)。
    C. 合乎逻辑的主体动作 (Logical Subject Action)： 描述角色或主体合乎情理的下一个动作。这个动作是叙事所必需的。
        关键词示例： The character stops reading, closes the book, and looks up... (角色停止阅读，合上书，然后抬头), The soldier raises their binoculars to scan the horizon... (士兵举起望远镜扫视地平线), The cat, having heard a noise, turns its head towards the door... (猫听到了噪音，转头望向门口), The hero lowers their shield after blocking the attack... (英雄在挡住攻击后放下了盾牌)。
    D. 合理的环境/大气变化 (Logical Environmental Change)： 描述一个符合叙事逻辑的环境变化。
        关键词示例： Rain begins to lightly fall, and the pavement starts to get wet... (开始下小雨，人行道开始变湿), The sun dips below the horizon, casting long shadows... (太阳落山，投下长长的影子), A second character walks into the room and stops... (第二个角色走进房间并停下), The door in the background opens... (背景里的门打开了)。

3，风格描述： 在提示词的末尾，添加标准的电影感风格化描述。
    关键词示例： Cinematic style (电影风格), realistic lighting (写实照明), clear composition (清晰的构图), natural colors (自然的色彩), good depth of field (良好的景深)。

输出格式示例（强调标准进展）：

Next Scene: The camera tracks alongside the protagonist as she walks briskly down the crowded city street. Cinematic style, natural lighting, urban atmosphere. (A 组合)
Next Scene: The shot cuts from the wide view to a close-up of the driver's eyes in the rear-view mirror. Tense mood, cinematic, shallow depth of field. (B 组合)
Next Scene: The character, who was listening intently, now nods slowly in agreement and begins to speak. Clear composition, soft lighting, realistic cinematic. (C 组合)
Next Scene: The camera holds steady on the house as the porch light flickers on, illuminating the front door. Cinematic, slightly moody lighting, clear focus. (D 组合)
Next Scene: The camera pulls back from the close-up of the map to a medium shot, showing the three adventurers studying it together. Realistic lighting, collaborative mood, cinematic. (A + C 组合)

你的任务：
请严格遵守以上所有规则，为我生成1个新的、变化标准且清晰的提示词。""",
            "图片指令-下一个镜头（强烈）": """你是一个专业的电影导演助理和 AI 图像生成提示词专家。你的任务是为 next-scene-qwen-image-lora-2509 LoRA 模型生成高质量的提示词。
你现在的核心任务是最大化变化。生成的提示词必须驱动 LoRA 产生一个与前一帧有显著视觉差异的镜头，必须像电影中一个关键的“新镜头”，而不是同一镜头的轻微延续。你要严格避免微妙的、微小的调整。

生成规则：

1，强制前缀： 每一个提示词都必须以英文短语 Next Scene: 开头（冒号后跟一个空格）。

2，核心内容（戏剧性转折）： 提示词的主体必须描述一个巨大且明显的动态变化。你必须从以下类别中选择，并使用强烈的、不容置疑的动词：
    A. 大幅度的摄像机运动与构图重塑 (Dramatic Camera Moves & Compositional Reshaping)：
        指令： 必须描述一个彻底改变构图的镜头运动。例如，从一个极端切换到另一个极端（广角到特写），或者一个横扫整个场景的宏大运镜。
        关键词示例： A rapid push-in from a wide shot to an extreme close-up on... (从广角快速推进到...的大特写), A sweeping crane shot that moves from ground-level up high, revealing... (一个从地面拉到高处的宏大摇臂镜头，揭示了...), The camera abruptly pulls back from the action to reveal a massive, unexpected threat... (镜头从动作中猛然后拉，揭示了一个巨大的、意外的威胁), The shot transitions from a subjective (POV) view to a detached high-angle shot. (镜头从主观视角切换到疏离的俯拍视角)。
    B. 决定性的主体动作 (Decisive Subject Action)：
        指令： 必须描述一个改变游戏规则的动作。这个动作必须在叙事上推进故事，而不仅仅是一个小姿势。
        关键词示例： The character, who was standing still, now breaks into a full sprint... (角色从静止变为全力冲刺), The protagonist suddenly ignites their powers, casting a blinding light... (主角突然点燃了他们的力量，发出耀眼的光芒), The dragon opens its mouth and unleashes a massive stream of fire... (龙张开嘴，喷射出巨大的火焰流), The character slams their fist on the table, scattering objects everywhere... (角色一拳砸在桌上，物品四处飞溅)。
    C. 剧烈的环境/大气突变 (Drastic Environmental/Atmospheric Shifts)：
        指令： 必须描述一个完全改变场景氛围或物理环境的事件。
        关键词示例： A sudden, violent sandstorm instantly engulfs the entire landscape... (一场突如其来的猛烈沙尘暴瞬间吞没了整个景观), The volcano in the background erupts, filling the sky with ash and lava... (背景中的火山爆发，火山灰和熔岩充满了天空), The clear sky instantly turns into a dark, supernatural tempest... (晴朗的天空瞬间变成了黑暗的、超自然的暴风雨), The entire scene is plunged into darkness as all lights suddenly go out, replaced by a single red emergency light. (所有灯光突然熄灭，整个场景陷入黑暗，取而代KAI的是一盏红色应急灯)。

3，强制组合（推荐）： 尽可能将类别 A、B 或 C 组合使用，以产生最大的视觉冲击力。（例如：一个摄像机移动 并且 一个主体动作）。       
        
4，风格描述： 在提示词的末尾，添加电影感的风格化描述，以支持这种戏剧性变化。
    关键词示例： Dramatic cinematic lighting (戏剧性的电影照明), high contrast (高对比度), epic scale (史诗规模), intense atmosphere (强烈的氛围), dynamic motion blur (动态模糊), wide-angle distortion (广角畸变)。

输出格式示例（强调明显变化）：

Next Scene: The camera rapidly pushes in from a wide shot of the battlefield to an extreme close-up on the general's face, capturing the exact moment he shouts "Charge!". Dramatic lighting, intense emotion, cinematic realism. (A + B 组合)
Next Scene: The camera pulls back dramatically from the character's hopeful expression to reveal that the entire city behind them is now in ruins. Epic scale, desolate atmosphere, high contrast cinematic. (A + C 组合)
Next Scene: Holding on the medium shot, the clear day suddenly turns to night as a massive alien mothership eclipses the sun, casting a terrifying shadow. Abrupt lighting change, ominous mood, cinematic sci-fi. (C 组合)
Next Scene: The camera tracks backwards quickly as the protagonist, who was cornered, suddenly leaps over the chasm towards the viewer. Dynamic motion blur, heroic pose, wide-angle lens, cinematic action. (A + B 组合)
Next Scene: The entire cave begins to collapse; the camera shakes violently as the character dives for cover under a large rock. Chaotic atmosphere, fast motion, dusty particles, realistic cinematic. (B + C 组合)

你的任务：
请严格遵守以上所有规则，为我生成1个新的、变化明显的提示词。""",
            "图片指令-下一个镜头（标准V2）": """你是一个专业的电影导演助理和 AI 图像生成提示词专家。你的任务是为 next-scene-qwen-image-lora-2509 LoRA 模型生成高质量的提示词。
你的目标是生成一个“标准的下一个镜头”。这个镜头必须与前一帧有清晰、明确、合乎逻辑的进展。

核心生成规则：

1，强制前缀： 每一个提示词都必须以英文短语 Next Scene: 开头（冒号后跟一个空格）。

2，核心内容（标准进展）： 提示词的主体必须清晰地描述一个“单一且明确”的变化。

3，!! 核心逻辑：主体-场景联动性 (Subject-Scene Correlation) !!
    此为最高规则： 你*绝不能*生成一个只有环境变化（如出现大坑）而主体（如人）保持完全不变（相同姿势、表情、位置）的提示词。
    因果关系： 主体的动作/表情/位置变化，与场景/摄像机的变化，**必须**有逻辑上的因果关系。
    （正确示例：） `Next Scene: A large chasm opens in the desert floor, causing the character to stumble backwards in surprise.` (环境变化 -> 角色反应)
    （正确示例：） `Next Scene: The character presses a button, and a holographic map appears in front of them.` (角色动作 -> 环境变化)
    （**错误**示例：） `Next Scene: A large chasm opens in the desert floor, and the character continues to stand there perfectly still.`

详细内容类别（必须遵守规则 3）：

A. 清晰的摄像机运动： 描述一个有目的的摄像机运动，通常是为了*跟随*一个动作或*揭示*一个反应。
    示例： `Next Scene: The camera tracks alongside the character as they begin to walk...` (摄像机跟随主体的动作)
    示例： `Next Scene: The camera pushes in on the character's face as they notice something off-screen.` (摄像机运动捕捉主体的反应)
B. 明确的剪辑切换： 描述一个新镜头，通常用于展示一个*反应镜头*或一个动作的*结果*。
    示例： `Next Scene: The shot cuts to a close-up of the character's hand as it reaches for the sword.` (剪辑以突出主体的动作)
    示例： `Next Scene: The shot cuts to a wide view, showing the character is now surrounded by wolves.` (剪辑以揭示环境变化，并暗示角色下一步的反应)
C. 合乎逻辑的主体动作 (因)： 描述角色或主体合乎情理的下一个动作，这个动作可能会*导致*环境变化。
    示例： `Next Scene: The character raises their hand, and the rocks in front of them begin to float.` (主体动作 -> 环境变化)
    示例： `Next Scene: The character turns their head towards the door as it begins to open.` (主体动作与环境变化同步发生)
D. 合理的环境变化 (果)： 描述一个环境变化，并*强制*主体对其做出反应。
    示例： `Next Scene: Rain begins to fall, forcing the character to pull up their hood.` (环境变化 -> 主体反应)
    示例： `Next Scene: The wall in front of the character crumbles, revealing a hidden passage and they step back in awe.` (环境变化 -> 主体反应)

风格描述： 在提示词的末尾，添加标准的电影感风格化描述。（例如：`Cinematic style`, `realistic lighting`, `clear composition`）

你的任务：
请严格遵守以上所有规则，特别是**规则 3 (主体-场景联动性)**，为我生成1个新的、**变化标准且逻辑严谨**的提示词。""",
            "图片指令-下一个镜头（强烈V2）": """你是一个专业的电影导演助理和 AI 图像生成提示词专家。你的任务是为 next-scene-qwen-image-lora-2509 LoRA 模型生成高质量的提示词。
你现在的核心任务是最大化变化，生成一个与前一帧有*显著*视觉差异的**关键“新镜头”**。

核心生成规则：

1，强制前缀： 每一个提示词都必须以英文短语 Next Scene: 开头（冒号后跟一个空格）。

2，核心内容（戏剧性转折）： 提示词的主体必须描述一个**巨大且明显的动态变化**。

3，!! 核心逻辑：戏剧性因果 (Dramatic Cause & Effect) !!
    此为最高规则： 这是“主体-场景联动性”的强化版。一个戏剧性的环境变化**必须**伴随一个戏剧性的主体反应。反之亦然。
    绝不脱节： 严禁出现“世界末日了，主角还在原地发呆”的逻辑错误。变化必须是*相互*的。
    （正确示例：） `Next Scene: The volcano erupts violently, forcing the character to dive for cover as ash and fire rain down.` (环境巨变 -> 主体剧烈反应)
    （正确示例：） `Next Scene: The hero slams their fist into the ground, causing the entire street to crack and shatter outwards.` (主体剧烈动作 -> 环境巨变)
    （**错误**示例：） `Next Scene: The volcano erupts violently, and the character just stands there looking.`

详细内容类别（必须遵守规则 3）：

A. 大幅度的摄像机运动： 描述一个彻底改变构图的镜头运动，用以*放大*一个动作或一个戏剧性后果。
    示例： `Next Scene: A rapid push-in from a wide shot to the character's terrified expression as the monster breaks through the wall.` (摄像机运动 + 主体反应 + 环境变化)
    示例： `Next Scene: A sweeping crane shot pulls up and away, revealing the lone character is now surrounded by a massive army, and they ready their stance.` (摄像机运动 + 环境揭示 + 主体反应)
B. 决定性的主体动作 (因)： 描述一个改变局面的动作，这个动作*导致*了戏剧性的后果。
    示例： `Next Scene: The protagonist ignites their superpowers, unleashing a blinding wave of energy that vaporizes the enemies in front of them.` (主体动作 -> 环境/他人变化)
    示例： `Next Scene: The character, previously cornered, suddenly leaps across the chasm in a desperate jump.` (主体戏剧性动作)
C. 剧烈的环境突变 (果)： 描述一个完全改变场景的事件，并*迫使*主体做出强烈的反应。
    示例： `Next Scene: A sudden, violent explosion to the left throws the character off their feet.` (环境巨变 -> 主体被动反应)
    示例： `Next Scene: The entire floor collapses, forcing the character to grab onto a ledge at the last second.` (环境巨变 -> 主体反应)
* 强制组合： 尽可能将 (B)主体动作 和 (C)环境变化 结合在一个提示词中，以创造最强烈的“因果”冲击力。

风格描述： 在提示词的末尾，添加支持戏剧性变化的风格化描述。（例如：`Dramatic cinematic lighting`, `high contrast`, `epic scale`, `dynamic motion blur`）

你的任务：
请严格遵守以上所有规则，特别是**规则 3 (戏剧性因果)**，为我生成1个新的、**变化明显且逻辑严谨**的提示词。""",
            "图片指令-图片打标": """You are an expert prompt engineer for advanced text-to-image models like SDXL and Pony. Analyze the image I've uploaded and generate a corresponding Positive Prompt and Negative Prompt based on the following rules.
Rules for Generation:

1, Format: All prompts must be in English, consisting of comma-separated keywords and short phrases.
2, Content First: Start the prompt with the core subject and composition of the image.
3, Layered Description: Structure the keywords in the order of "Subject > Details > Style > Composition & Lighting > Quality".
4, Identify Style & IP: Accurately identify all characters, intellectual properties (like movie or game titles), artist styles, and the overall aesthetics in the image.
5, Quality Tags: Include universal high-quality tags like masterpiece, best quality at the beginning or end. For Pony models, add scoring tags like score_9, score_8_up.
6, Output Structure: Strictly follow the format below, without any extra explanations or prose.

【Positive Prompt】
[Generate Positive Prompt here]

【Negative Prompt】
[Generate Negative Prompt here]""",
            "图片指令-首尾过度（Wan）": "You are a cinematic prompt composer for the wan2.2 video generation model. You are given a composite image containing two frames: the right side is the starting frame of the video, and the left side is the ending frame. Your task is to analyze both frames and imagine a smooth, emotionally coherent transition between them. Generate a single, detailed video prompt that describes the full temporal arc—from the initial pose to the final pose—by constructing plausible intermediate actions, gestures, expressions, and environmental changes. Include subject movement, camera behavior, framing, lighting evolution, and pacing. The transition must feel natural and continuous, without abrupt jumps. Do not explain your reasoning—just output the final prompt.",
            "图片指令-图片描述（Wan）": "You are a cinematic prompt composer for the wan2.2 video generation model. Based on the uploaded image, analyze its visual content and extrapolate a fully detailed video prompt. Your output must incorporate accurate and creative inferences of (1) subject appearance and identity, (2) setting and environmental design, (3) motion or action, (4) cinematic composition and camera behavior, (5) lighting and mood, and (6) stylization and aesthetic tone. Frame the output as a dynamic scene suitable for temporal video generation. Use vivid imagery, expressive verbs, and avoid static descriptions. Highlight subtle emotional nuance and visual storytelling potential. Do not over-describe static elements—favor dynamic and expressive language. Do not add any conversational text, explanations, or deviations",
            "图片指令-图片颜色": "Analyze the image and extract the plain_english_colors of the 5 main colors, separated by commas",
            "图片指令-图片HEX": "Analyze the image and extract the HEX values of the 5 main colors, separated by commas",
            "图片指令-图片RGB": "Analyze the image and extract the RGB values of the 5 main colors, separated by commas and parentheses",
            "视频指令-视频描述": "Describe the video in detail and accurately",
            "视频指令-默片补音": "You are a voice synthesis prompt composer for the MMAudio model. Your task is to generate realistic and expressive voice-based audio for a silent video. Analyze the visual content frame by frame, and for each segment, determine the appropriate sound—whether it's speech, ambient noise, mechanical sounds, emotional vocalizations, or symbolic utterances. Your output must reflect the timing, pacing, and emotional tone of the video, matching visual actions and transitions precisely. Avoid vague or generic words; instead, write full, meaningful voice or sound content that would naturally accompany the visuals. Do not add any conversational text, explanations, deviations, or numbering.",
            "音频指令-音频描述": "Describe the audio in detail and accurately"
        }
        
        kontext_prompt_map = {
            "无指令": "",
            "图片指令-图片编辑-传送": instruction_base_single + "Teleport the subject to a random location, scenario and/or style. Re-contextualize it in various scenarios that are completely unexpected. Do not instruct to replace or transform the subject, only the context/scenario/style/clothes/accessories/background..etc." + common_suffix_single,
            "图片指令-图片编辑-移动相机": instruction_base_single + "Move the camera to reveal new aspects of the scene. Provide highly different types of camera mouvements based on the scene (eg: the camera now gives a top view of the room; side portrait view of the person..etc )." + common_suffix_single,
            "图片指令-图片编辑-重打光": instruction_base_single + "Suggest new lighting settings for the image. Propose various lighting stage and settings, with a focus on professional studio lighting.\n\nSome suggestions should contain dramatic color changes, alternate time of the day, remove or include some new natural lights…etc" + common_suffix_single,
            "图片指令-图片编辑-重构图": instruction_base_single + "Recompose the entire image while keeping the core subjects. Suggest a new framing, object placement, and visual hierarchy to improve aesthetic flow." + common_suffix_single,
            "图片指令-图片编辑-产品摄影": instruction_base_single + "Turn this image into the style of a professional product photo. Describe a variety of scenes (simple packshot or the item being used), so that it could show different aspects of the item in a highly professional catalog.\n\nSuggest a variety of scenes, light settings and camera angles/framings, zoom levels, etc.\n\nSuggest at least 1 scenario of how the item is used." + common_suffix_single,
            "图片指令-图片编辑-产品外观": instruction_base_single + "You are a visual product designer. Modify the outer appearance of the product in the image while retaining its core shape and purpose. Suggest a new stylistic direction—such as luxury packaging, minimalist tech casing, eco-friendly wrapping, retro appliance housing, or futuristic industrial look. Update color palette, materials, surface textures, and visible branding elements to match the new design intent." + common_suffix_single,
            "图片指令-图片编辑-缩放": instruction_base_single + "Analyze the image and zoom on its main subject. Provide different levels of zoom. If an explicit subject is provided within the image, focus on that; otherwise, determine the most prominent subject. Always provide diverse zoom levels.\n\nExample: Zoom on the abstract painting above the fireplace to focus on its details, capturing the texture and color variations, while slightly blurring the surrounding room for a moderate zoom effect." + common_suffix_single,
            "图片指令-图片编辑-上色": instruction_base_single + "Colorize the image. Provide different color styles / restoration guidance." + common_suffix_single,
            "图片指令-图片编辑-LOGO设计": instruction_base_single + "You are a visual identity designer. Transform the main object or character in the image into a stylized logo representation. Reduce complexity while preserving recognizable features, using abstracted forms, simplified shapes, and strong contrast. Apply flat color palette or vector aesthetics typical of professional logos, such as monochrome, minimal line work, or bold emblem style. The result should be scalable and symbolically representative, suitable for brand use." + common_suffix_single,
            "图片指令-图片编辑-尾帧图像": instruction_base_single + "You are a cinematic sequence planner. Analyze the original image and generate a final frame that complements the initial frame for high-quality video interpolation. Examine subject pose, environment, motion cues, and narrative context to infer what a compelling ending frame would look like—whether it's a resolved action, a visual climax, a spatial transition, or an emotional evolution. Ensure the tail frame introduces enough visual transformation while maintaining semantic coherence to guide the video model effectively." + common_suffix_single,
            "图片指令-图片编辑-电影海报": instruction_base_single + "Create a movie poster with the subjects of this image as the main characters. Take a random genre (action, comedy, horror, etc) and make it look like a movie poster.\n\nGenerate a stylized movie title based on the image content or infer one if clearly intended, and add relevant taglines. Also, include various textual elements typically seen in movie posters like quotes or credits." + common_suffix_single,
            "图片指令-图片编辑-卡通化": instruction_base_single + "Turn this image into the style of a cartoon or manga or drawing. Include a reference of style, culture or time (eg: mangas from the 90s, thick lined, 3D pixar, etc)" + common_suffix_single,
            "图片指令-图片编辑-移除文字": instruction_base_single + "Remove all text from the image." + common_suffix_single,
            "图片指令-图片编辑-发型更换": instruction_base_single + "Change only the haircut of the subject. Suggest a new hairstyle with details such as cut, texture, length, and color, but preserve the rest of the image strictly unchanged. Do not modify face structure, facial expression, clothing, pose, body proportions, or the background. Ensure the new hairstyle integrates naturally on the original head position, respecting lighting and silhouette alignment. This is a localized transformation." + common_suffix_single + local_edit_suffix,
            "图片指令-图片编辑-服装更换": instruction_base_single + "Change the subject’s clothing into a different style (e.g., traditional Hanfu, cyberpunk gear, royal garments). Maintain natural integration with pose and lighting." + common_suffix_single + local_edit_suffix,
            "图片指令-图片编辑-写真达人": instruction_base_single + "You are a portrait pose director. Based on the subject's inherent aura and the ambient context in the image, construct a soft idol-style composition that enhances visual rhythm and emotional subtlety. Freely select an imaginative character identity and evocative scene atmosphere that suits the source—whether cinematic, retro, serene, stylish, ethereal, or abstract. Ensure the final pose radiates intimacy, elegance, or soft allure through finely tuned body language. Every limb angle, hand gesture, gaze direction, and emotional suggestion must be precise and coherent. Maintain strict fidelity to facial structure using facial ID locking; identity integrity must be preserved throughout." + common_suffix_single,
            "图片指令-图片编辑-健身达人": instruction_base_single + "Ask to largely increase the muscles of the subjects while keeping the same pose and context.\n\nDescribe visually how to edit the subjects so that they turn into bodybuilders and have these exagerated large muscles: biceps, abdominals, triceps, etc.\n\nYou may change the clothse to make sure they reveal the overmuscled, exagerated body." + common_suffix_single,
            "图片指令-图片编辑-移除家具": instruction_base_single + "Remove all furniture and all appliances from the image. Explicitely mention to remove lights, carpets, curtains, etc if present." + common_suffix_single,
            "图片指令-图片编辑-室内设计": instruction_base_single + "You are an interior designer. Redo the interior design of this image. Imagine some design elements and light settings that could match this room and offer diverse artistic directions, while ensuring that the room structure (windows, doors, walls, etc) remains identical." + common_suffix_single,
            "图片指令-图片编辑-建筑外观": instruction_base_single + "You are a visual architect. Modify the external appearance of the building in the image while maintaining its core structure (walls, roof, entrances, windows). Suggest a new architectural style or facade design (e.g., modern minimalist, classical European, traditional East Asian, cyberpunk skyscraper). Adjust texture, materials, colors, and ornamental details accordingly to ensure natural integration with surroundings." + common_suffix_single,
            "图片指令-图片编辑-清理杂物": instruction_base_single + "You are a cleanup editor. Your mission is to detect and remove all visual clutter or unnecessary objects from the image. These cluttered elements may include random items, debris, scattered objects, cables, packaging, trash bins, storage piles, or background distractions. Ensure removal enhances the aesthetic clarity and does not affect core subjects or intentional props." + common_suffix_single,
            "图片指令-图片编辑-艺术风格": instruction_base_single + "Repaint the entire image in the unmistakable style of a famous art movement. Be descriptive and evocative. For example: 'Transform the image into a Post-Impressionist painting in the style of Van Gogh, using thick, swirling brushstrokes (impasto), vibrant, emotional colors, and a dynamic sense of energy and movement in every element.'" + common_suffix_single,
            "图片指令-图片编辑-材质转换": instruction_base_single + "Transform the material of a specific object or surface (e.g., wood to glass, cloth to metal). Indicate visual cues for texture, reflectivity, and interaction with light." + common_suffix_single,
            "图片指令-图片编辑-情绪变化": instruction_base_single + "Alter the emotional tone of the subject in the image. Suggest a clear transformation of the facial expression, body language, and atmosphere to convey a new emotion (e.g., turn neutral to joyous, anxious to serene). Maintain realism and natural transition." + common_suffix_single,
            "图片指令-图片编辑-年龄变化": instruction_base_single + "Visibly and realistically age or de-age the main subject, representing a different life stage. Describe detailed facial and hair changes while maintaining core features." + common_suffix_single,
            "图片指令-图片编辑-季节变化": instruction_base_single + "Transform the scene’s season (e.g., from spring to winter or summer to autumn). Modify foliage, lighting, clothing, and ambient atmosphere to reflect the new season." + common_suffix_single,
            "图片指令-图片编辑-合成融合": instruction_base_single + "Detect any compositing mismatches between the foreground subject and background, such as inconsistent lighting, shadows, color grading, or depth cues. Perform automatic fusion to harmonize both elements by adjusting lighting direction, shadow placement, color tones, perspective alignment, and edge blending to ensure seamless realism." + common_suffix_single,
            "图片指令-图片编辑-调色板": """根据下面的几种颜色来对这张图像的色彩进行重新设计图像编辑提示词，比如把图像中的某个物体换成某个颜色，确保每一种颜色都被应用到图像上，你只需要输出如下格式的示例文本：
将“天空”的颜色改为“dark sky blue”。
将女孩T恤上的“蓝色领口”和数字“83”的颜色改为“dark indigo”。
将“泥土小径”的颜色改为“golden yellow”。
将女孩的“红色裙子”颜色改为“bright violet”。
将“小羊的鼻子和内耳”以及女孩“羊帽上的鼻子”的颜色，改为“brownish pink”。

最后你需要使用的颜色如下：""",
        }

        kontext_prompt_no_api_map = {
            "无指令": "",
            "角色编辑 - 表情": "Change the facial expression of the person in the image to {prompt} while keeping the head orientation, facial structure, scale, and lighting exactly the same. Maintain identical placement, pose, and all other features. Only update the expression by adjusting mouth shape, eyebrows, and eye tension to match the intended emotion, without modifying the rest of the face or body.",
            "角色编辑 - 动作": "Change the pose of the person in the image to {prompt}, while keeping their identity, face, hairstyle, scale, and clothing exactly the same. Maintain identical camera angle, lighting, and environmental context. Only update the body posture and gesture based on the target pose description, ensuring consistent anatomy, joint placement, and perspective.",
            "角色编辑 - 皮肤": "Replace only the commonly exposed skin areas (face, neck, arms, and hands) of the person in the image with {prompt}, while keeping the pose, body proportions, facial structure, and expression exactly the same. Maintain identical placement, camera angle, lighting, and all other visual features. Do not modify clothing, accessories, or any non-biological surfaces. Clothing must retain its original material and color. Remove biological details or residual markings from the affected skin regions.",
        }

        if preset_prompt != "无指令":
            prompt = f"{preset_prompt_map.get(preset_prompt, '')}. {prompt}"
        
        if kontext_prompt != "无指令":
            prompt = f"{kontext_prompt_map.get(kontext_prompt, '')}. {prompt}"

        if kontext_prompt_no_api != "无指令":
            template = kontext_prompt_no_api_map.get(kontext_prompt_no_api, "")
            if template and "{prompt}" in template:
                prompt = template.format(prompt=prompt)
            elif template:
                prompt = f"{template}. {prompt}"

        if model_version == "no-api":
            return (prompt, self.create_placeholder_image())

        safety_settings = [
            {"category": "harassment", "threshold": "NONE"},
            {"category": "hate_speech", "threshold": "NONE"},
            {"category": "sexually_explicit", "threshold": "NONE"},
            {"category": "dangerous_content", "threshold": "NONE"},
            {"category": "civic", "threshold": "NONE"}
        ]

        if api_key.strip():
            self.api_key = api_key
            save_config({"GEMINI_API_KEY": self.api_key})
            self.configure_genai()

        if not self.api_key:
            raise ValueError("API key not found in config.json or node input")

        if clear_history:
            self.chat_history.clear()

        if operation_mode == "generate_images":
            return self.generate_images(
                prompt=prompt,
                model_version=model_version,
                images=images,
                batch_count=batch_count,
                temperature=temperature,
                seed=seed,
                max_images=max_images
            )

        model_name = f'models/{model_version}'
        model = genai.GenerativeModel(model_name)

        model.safety_settings = safety_settings

        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature
        )

        try:
            if chat_mode:
                if input_type == "text":
                    text_content = prompt if not Additional_Context else f"{prompt}\n{Additional_Context}"
                    content = text_content
                elif input_type == "image":
                    all_images = []
                    
                    if images is not None:
                        if isinstance(images, torch.Tensor) and len(images.shape) == 4:
                            num_to_process = min(images.shape[0], max_images)
                            for i in range(num_to_process):
                                pil_image = self.tensor_to_image(images[i])
                                pil_image = self.resize_image(pil_image, 1024)
                                all_images.append(pil_image)
                        elif isinstance(images, list):
                            for img_tensor in images[:max_images]:
                                pil_image = self.tensor_to_image(img_tensor)
                                pil_image = self.resize_image(pil_image, 1024)
                                all_images.append(pil_image)
                    
                    if all_images:
                        img_count = len(all_images)
                        prefix = f"Analyzing {img_count} image{'s' if img_count > 1 else ''}. "
                        if img_count > 1:
                            prefix += "Please describe each image separately. "
                        
                        parts = [{"text": f"{prefix}{prompt}"}]
                        
                        for img in all_images:
                            img_byte_arr = BytesIO()
                            img.save(img_byte_arr, format='PNG')
                            img_bytes = img_byte_arr.getvalue()
                            
                            parts.append({
                                "inline_data": {
                                    "mime_type": "image/png", 
                                    "data": base64.b64encode(img_bytes).decode('utf-8')
                                }
                            })
                        
                        content = {"parts": parts}
                    else:
                        raise ValueError("No images provided for image input type")
                elif input_type == "video" and video is not None:
                    if len(video.shape) == 4 and video.shape[0] > 1:
                        frame_count = video.shape[0]
                        frames = self.sample_video_frames(video)
                        if frames:
                            parts = [{"text": f"This is a video with {frame_count} frames. {prompt}"}]
                            
                            for frame in frames:
                                img_byte_arr = BytesIO()
                                frame.save(img_byte_arr, format='PNG')
                                img_bytes = img_byte_arr.getvalue()
                                
                                parts.append({
                                    "inline_data": {
                                        "mime_type": "image/png",
                                        "data": base64.b64encode(img_bytes).decode('utf-8') 
                                    }
                                })
                            
                            content = {"parts": parts}
                        else:
                            raise ValueError("Error processing video frames")
                    else:
                        pil_image = self.tensor_to_image(video.squeeze(0) if len(video.shape) == 4 else video)
                        pil_image = self.resize_image(pil_image, 1024)
                        
                        img_byte_arr = BytesIO()
                        pil_image.save(img_byte_arr, format='PNG')
                        img_bytes = img_byte_arr.getvalue()
                        
                        content = {"parts": [
                            {"text": f"This is a single frame from a video. {prompt}"},
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": base64.b64encode(img_bytes).decode('utf-8')
                                }
                            }
                        ]}
                elif input_type == "audio" and audio is not None:
                    waveform = audio["waveform"]
                    sample_rate = audio["sample_rate"]
                    
                    if waveform.dim() == 3:
                        waveform = waveform.squeeze(0)
                    elif waveform.dim() == 1:
                        waveform = waveform.unsqueeze(0)
                    
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    if sample_rate != 16000:
                        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
                    
                    buffer = BytesIO()
                    torchaudio.save(buffer, waveform, 16000, format="WAV")
                    audio_bytes = buffer.getvalue()
                    
                    content = {"parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "audio/wav",
                                "data": base64.b64encode(audio_bytes).decode('utf-8')
                            }
                        }
                    ]}
                else:
                    raise ValueError(f"Invalid or missing input for {input_type}")

                chat = model.start_chat(history=self.chat_history.get_messages_for_api())
                response = chat.send_message(content, generation_config=generation_config)
                
                if isinstance(content, dict) and "parts" in content:
                    history_content = prompt
                else:
                    history_content = content
                    
                self.chat_history.add_message("user", history_content)
                self.chat_history.add_message("assistant", response.text)

                generated_content = self.chat_history.get_formatted_history()
            else:
                content_parts = self.prepare_content(
                    prompt, input_type, Additional_Context, images, video, audio, max_images
                )
                
                if structured_output:
                    if isinstance(content_parts, list) and len(content_parts) > 0:
                        if "parts" in content_parts[0]:
                            for part in content_parts[0]["parts"]:
                                if "text" in part:
                                    part["text"] = f"Please provide the response in a structured format. {part['text']}"
                
                response = model.generate_content(content_parts, generation_config=generation_config)
                generated_content = response.text

        except Exception as e:
            generated_content = f"Error: {str(e)}"
    
        return (generated_content, self.create_placeholder_image())
        
NODE_CLASS_MAPPINGS = {
    "GeminiFlash": GeminiFlash,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiFlash": "Gemini 提示词多功能版",
}