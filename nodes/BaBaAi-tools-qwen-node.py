import os
import uuid
import folder_paths
import numpy as np
import re
import random
import time
from collections import Counter
import hashlib
import io

from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    BitsAndBytesConfig,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer
)
from pathlib import Path
from comfy_api.input import VideoInput
import torch

class CustomTextStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=True):
        super().__init__(tokenizer, skip_prompt=skip_prompt, skip_special_tokens=skip_special_tokens)
        self.generated_text = ""
        self.stop_flag = False
        self.init_time = time.time()
        self.end_time = None
        self.first_token_time = None
        self.token_count = 0

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if self.first_token_time is None and text.strip():
            self.first_token_time = time.time()
        self.generated_text += text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        self.token_count += len(tokens)
        
        if stream_end:
            self.end_time = time.time()
        if self.stop_flag:
            raise StopIteration

    def stop_generation(self):
        self.stop_flag = True
        self.end_time = time.time()

    def get_metrics(self):
        if self.end_time is None:
            self.end_time = time.time()
        total_time = self.end_time - self.init_time
        tokens_per_second = self.token_count / total_time if total_time > 0 else 0
        first_token_latency = (self.first_token_time - self.init_time) if self.first_token_time is not None else None
        
        metrics = {
            "init_time": self.init_time,
            "first_token_time": self.first_token_time,
            "first_token_latency": first_token_latency,
            "end_time": self.end_time,
            "total_time": total_time,
            "total_tokens": self.token_count,
            "tokens_per_second": tokens_per_second
        }
        
        metrics_str = f"Metrics:\n"
        for key, value in metrics.items():
            metrics_str += f"  {key}: {value}\n"
        return metrics_str

def temp_video(video: VideoInput, seed):
    unique_id = uuid.uuid4().hex
    video_path = Path(folder_paths.temp_directory) / f"temp_video_{seed}_{unique_id}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video.save_to(os.path.join(video_path), format="mp4", codec="h264")
    return video_path.as_posix()

def temp_image(image, seed):
    unique_id = uuid.uuid4().hex
    image_path = Path(folder_paths.temp_directory) / f"temp_image_{seed}_{unique_id}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    img.save(os.path.join(image_path))
    return image_path.as_posix()

def temp_batch_image(image, num_counts, seed):
    image_batch_path = Path(folder_paths.temp_directory) / "Multiple"
    image_batch_path.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for Nth_count in range(num_counts):
        img = Image.fromarray(np.clip(255.0 * image[Nth_count].cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        unique_id = uuid.uuid4().hex
        image_path = image_batch_path / f"temp_image_{seed}_{Nth_count}_{unique_id}.png"
        img.save(os.path.join(image_path))
        image_paths.append(image_path.resolve().as_posix())
    return image_paths

model_directory = os.path.join(folder_paths.models_dir, "VLM")
os.makedirs(model_directory, exist_ok=True)

class Qwen_VL_Model_Loader_Base:
    def load_model(self, model_class, model_repo_id, quantization, attention):
        model_dict = {"model": "", "model_path": ""}
        model_name = model_repo_id.rsplit("/", 1)[-1]
        model_path = os.path.join(model_directory, model_name)
        
        if not os.path.exists(model_path):
            print(f"Downloading {model_name} to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_repo_id, local_dir=model_path, local_dir_use_symlinks=False)

        quant_config = None
        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True
            )
        elif quantization == "8bit":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )

        print(f"[BaBaAi-Tools] Attempting to load model with '{attention}' attention...")

        model_instance = model_class.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            attn_implementation=attention,
            quantization_config=quant_config,
            trust_remote_code=True,
        )

        effective_attn = getattr(model_instance.config, "_attn_implementation", "unknown")
        print(f"[BaBaAi-Tools] Model '{model_name}' loaded successfully.")
        print(f"[BaBaAi-Tools] >>> Effective Attention Implementation: {effective_attn.upper()} <<<")
        if attention == "flash_attention_2" and effective_attn != "flash_attention_2":
            print("[BaBaAi-Tools] WARNING: FlashAttention 2 was requested but is not active. The model has fallen back to a different implementation. Check your GPU, CUDA, and flash-attn installation.")

        model_dict["model"] = model_instance
        model_dict["model_path"] = model_path
        
        if "Qwen3" in model_class.__name__:
            model_dict["processor"] = AutoProcessor.from_pretrained(model_path)
            
        return (model_dict,)

class Qwen2_5_VLModelLoader(Qwen_VL_Model_Loader_Base):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (["unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit", "huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated"], {"default": "huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated"}),
            "quantization": (["none", "4bit", "8bit"], {"default": "4bit"}),
            "attention": (["flash_attention_2", "sdpa", "eager"], {"default": "flash_attention_2"}),
        }}
    RETURN_TYPES = ("QWEN_VL_MODEL",)
    RETURN_NAMES = ("Qwen_VL_model",)
    FUNCTION = "load_2_5"
    CATEGORY = "📜BaBaAi Tools"
    def load_2_5(self, model, quantization, attention):
        return self.load_model(Qwen2_5_VLForConditionalGeneration, model, quantization, attention)

class Qwen3_VL_ModelLoader(Qwen_VL_Model_Loader_Base):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (["unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit", "huihui-ai/Huihui-Qwen3-VL-2B-Instruct-abliterated", "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated"], {"default": "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated"}),
            "quantization": (["none", "4bit", "8bit"], {"default": "4bit"}),
            "attention": (["flash_attention_2", "sdpa", "eager"], {"default": "flash_attention_2"}),
        }}
    RETURN_TYPES = ("QWEN_VL_MODEL",)
    RETURN_NAMES = ("Qwen_VL_model",)
    FUNCTION = "load_3"
    CATEGORY = "📜BaBaAi Tools"
    def load_3(self, model, quantization, attention):
        return self.load_model(Qwen3VLForConditionalGeneration, model, quantization, attention)

class Qwen_VL_Run_Advanced:
    instruction_base_single = "You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.\n\nThe brief:\n\n"
    common_suffix_single = "\n\nYour response must be a single, complete, and concise instruction ready for the image editing AI. Do not add any conversational text, explanations, deviations, or numbering."
    local_edit_suffix = "Finally, be sure to add the following after the prompt: This transformation must only affect the specified regio, Do not modify facial features, body structure, background, pose, lighting, or any other part of the image, Changes must be localized and preserve global consistency."
    _PRESET_PROMPT_MAP = {
        "无指令": "",
        "文本指令-提示词增强（NSFW）": "Generate a concise, explicit, and vivid enhancement for this text prompt, transforming it into a sharply detailed erotic scene optimized for AI image generation. Expand only on core suggested elements with sexual or intense interactions, using raw, immersive language from adult erotica—like 'throbbing cock slamming into her dripping pussy,' 'sweat-glistened tits bouncing mid-fuck,' or 'cum-splattered face in ecstasy'—to amp up the heat without excess. Match the vulgarity to the prompt's intensity: build sensual tension from tame hints like a director framing a porn still, avoiding unrelated extremes or gratuitous additions. Never sanitize explicit cues, but don't bloat neutral details into filth. Tune the tone to the prompt's essence—sultry tease, savage dominance, or fetish edge—and spotlight vivid visuals like arched backs in mid-thrust, torn lingerie clinging to curves, gritted teeth in orgasmic snarl, and glistening fluids on taut skin. Limit output to 150-200 words max for punchy, focused prompts; keep it evocative and precise, ditching vague fluff for hyper-specific, frame-ready eroticism.",
        "文本指令-提示词增强（Flux）": "You are a prompt enhancer for the Flux image generation model. Given a short Chinese phrase describing any subject—person, animal, object, or scene—translate it into English and expand it into a single, richly detailed and visually evocative prompt. By default, generate descriptions in the style of realistic photography, including details such as subject appearance, pose, clothing or texture, environment, lighting, mood, and camera perspective. If the original phrase explicitly mentions an artistic style (e.g., flat illustration, oil painting, cyberpunk), adapt the description to match that style instead. Do not provide multiple options or explanations—just output one complete, vivid paragraph suitable for high-quality image synthesis.",
        "文本指令-提示词增强（Flux中文）": "You are a prompt enhancement assistant for the Flux image generation model. Users will enter a short Chinese phrase, possibly describing a person, animal, object, or scene. Please expand this phrase into a complete, detailed, and visually expressive Chinese prompt. By default, please describe the subject in a realistic photographic style, including details such as the subject's appearance, pose, clothing or material, environment, lighting, atmosphere, and camera angle. If the original phrase explicitly mentions an artistic style (such as flat illustration, oil painting, cyberpunk, etc.), please construct the prompt based on that style. Do not provide multiple options or explanations, and only output a complete Chinese prompt suitable for high-quality image generation.",
        "文本指令-提示词增强（标签）": "Expand the following input into tag-style phrases suitable for Stable Diffusion 1.5 or SDXL. The output must consist of short, descriptive English keywords only—no full sentences or natural language. Follow this structure and order: Character Features (e.g., age, gender, appearance, clothing), Environment (e.g., background, setting), Lighting (e.g., time of day, light type), View Angle (e.g., close-up, wide shot, top-down), and Image Quality (e.g., high quality, 8k, cinematic). If the original input is in Chinese, translate it into English first. Only generate one set of tags—do not provide multiple options or explanations.",
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
        "图片指令-图片描述": "Describe the image in detail and accurately",
        "图片指令-图片描述（中文）": "详细描叙这张图片的内容。输出中文提示词。",
        "图片指令-图片描述（NSFW）": "Generate an explicit and vivid caption for this image that accurately describes all visible elements, including sexual or violent actions. Use intense language typical of adult themes, incorporating terms like 'fucking,' 'gets fucked,' 'dick,' 'pussy,' 'cum,' or other appropriate words to the descripe. But match the intensity of your description to what is actually shown, like porn-director or film-director. Don't sanitize explicit content, but also don't make innocent content sound more vulgar than it is. Ensure the tone aligns with the image's content—whether sensual, aggressive, or fetishistic—and highlight specific details such as body positions, clothing, facial expressions, and any explicit acts. Maintain clarity and avoid vague terms.",
        "图片指令-图片打标": "Use tags to briefly and accurately describe this image, separated by commas, don’t use duplicate tags",
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
        "图片指令-首尾过度（Wan）": "You are a cinematic prompt composer for the wan2.2 video generation model. You are given a composite image containing two frames: the right side is the starting frame of the video, and the left side is the ending frame. Your task is to analyze both frames and imagine a smooth, emotionally coherent transition between them. Generate a single, detailed video prompt that describes the full temporal arc—from the initial pose to the final pose—by constructing plausible intermediate actions, gestures, expressions, and environmental changes. Include subject movement, camera behavior, framing, lighting evolution, and pacing. The transition must feel natural and continuous, without abrupt jumps. Do not explain your reasoning—just output the final prompt.",
        "图片指令-图片描述（Wan）": "You are a cinematic prompt composer for the wan2.2 video generation model. Based on the uploaded image, analyze its visual content and extrapolate a fully detailed video prompt. Your output must incorporate accurate and creative inferences of (1) subject appearance and identity, (2) setting and environmental design, (3) motion or action, (4) cinematic composition and camera behavior, (5) lighting and mood, and (6) stylization and aesthetic tone. Frame the output as a dynamic scene suitable for temporal video generation. Use vivid imagery, expressive verbs, and avoid static descriptions. Highlight subtle emotional nuance and visual storytelling potential. Do not over-describe static elements—favor dynamic and expressive language. Do not add any conversational text, explanations, or deviations",
        "图片指令-图片颜色": "Analyze the image and extract the plain_english_colors of the 5 main colors, separated by commas",
        "图片指令-图片HEX": "Analyze the image and extract the HEX values of the 5 main colors, separated by commas",
        "图片指令-图片RGB": "Analyze the image and extract the RGB values of the 5 main colors, separated by commas and parentheses",
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
        "图片指令-图片编辑-调色板": """根据下面的几种颜色来对这张图像的色彩进行重新设计图像编辑提示词，比如把图像中的单个或者多个物体换成某个颜色，确保每一种颜色都被应用到图像上，需要考虑图像中的所有因素，不能仅针对主体进行颜色替换。输出范本为：
将“某物体”的颜色改为“某颜色”。

注意上面只是给你参考的输出格式样本，你需要分析我提供的图像来重新设计，颜色更换提示词限制在5至10行内，同时需要注意避免对同一物体进行多次颜色修改，强调“把某物体的颜色换成某种颜色”，而不是“把某物体换成某种颜色”。最后你需要使用的颜色如下：""",
        "视频指令-视频描述": "Describe the video in detail and accurately",
        "视频指令-默片补音": "You are a voice synthesis prompt composer for the MMAudio model. Your task is to generate realistic and expressive voice-based audio for a silent video. Analyze the visual content frame by frame, and for each segment, determine the appropriate sound—whether it's speech, ambient noise, mechanical sounds, emotional vocalizations, or symbolic utterances. Your output must reflect the timing, pacing, and emotional tone of the video, matching visual actions and transitions precisely. Avoid vague or generic words; instead, write full, meaningful voice or sound content that would naturally accompany the visuals. Do not add any conversational text, explanations, deviations, or numbering.",
        "音频指令-音频描述": "Describe the audio in detail and accurately"
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {"image": ("IMAGE",), "video": ("VIDEO",)},
            "required": {
                "Qwen_VL_model": ("QWEN_VL_MODEL",),
                "system_text": ("STRING", {"default": "Text output is limited to 200 words", "multiline": True}),
                "preset_prompt": (list(cls._PRESET_PROMPT_MAP.keys()), {"default": "无指令"}),
                "text": ("STRING", {"default": "", "multiline": True}),
                "video_decode_method": (["torchvision", "decord", "torchcodec"], {"default": "torchvision"}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100}),
                "min_pixels": ("INT", {"default": 256, "min": 64, "max": 1280}),
                "max_pixels": ("INT", {"default": 1280, "min": 64, "max": 2048}),
                "total_pixels": ("INT", {"default": 20480, "min": 1, "max": 24576}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "📜BaBaAi Tools"

    def run(self, Qwen_VL_model, system_text, preset_prompt, text, video_decode_method, max_new_tokens, temperature, top_p, top_k, min_pixels, max_pixels, total_pixels, seed, image=None, video=None):
        from qwen_vl_utils import process_vision_info

        model_dict = Qwen_VL_model
        model = model_dict["model"]
        processor = model_dict.get("processor") or AutoProcessor.from_pretrained(model_dict["model_path"])

        min_pixels *= 28 * 28
        max_pixels *= 28 * 28
        total_pixels *= 28 * 28
        
        full_text = f"{self._PRESET_PROMPT_MAP.get(preset_prompt, '')}\n{text}" if preset_prompt != "无指令" else text

        content = []
        if image is not None:
            num_counts = image.shape[0]
            if num_counts == 1:
                uri = temp_image(image, seed)
                content.append({"type": "image", "image": uri, "min_pixels": min_pixels, "max_pixels": max_pixels})
            else:
                image_paths = temp_batch_image(image, num_counts, seed)
                for path in image_paths:
                    content.append({"type": "image", "image": path, "min_pixels": min_pixels, "max_pixels": max_pixels})

        if video is not None:
            uri = temp_video(video, seed)
            content.append({"type": "video", "video": uri, "min_pixels": min_pixels, "max_pixels": max_pixels, "total_pixels": total_pixels})

        if full_text.strip():
            content.append({"type": "text", "text": full_text})

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": content},
        ]
        
        modeltext = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        os.environ["FORCE_QWENVL_VIDEO_READER"] = video_decode_method
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        if video is not None: 
            default_fps = 30.0
            final_fps = default_fps

            if "videos_kwargs" in video_kwargs and isinstance(video_kwargs.get("videos_kwargs"), dict):
                if "fps" in video_kwargs["videos_kwargs"]:
                    fps_value = video_kwargs["videos_kwargs"]["fps"]
                    
                    print(f"[BaBaAi-Tools DEBUG] Original fps_value type: {type(fps_value)}, value: {repr(fps_value)}") 

                    try:
                        if isinstance(fps_value, (float, int)):
                            final_fps = float(fps_value)
                        elif isinstance(fps_value, (list, tuple)):
                            temp_fps = None
                            for item in fps_value:
                                if isinstance(item, (float, int)):
                                    temp_fps = float(item)
                                    break
                                elif isinstance(item, (list, tuple)): 
                                    for sub_item in item:
                                         if isinstance(sub_item, (float, int)):
                                             temp_fps = float(sub_item)
                                             break
                                if temp_fps is not None:
                                    break
                            
                            if temp_fps is not None:
                                final_fps = temp_fps
                            else:
                                print(f"[BaBaAi-Tools WARNING] fps_value is a sequence but contains no number or is empty: {repr(fps_value)}. Using default FPS.")
                        else:
                            print(f"[BaBaAi-Tools DEBUG] Attempting direct float conversion for non-numeric/non-sequence type.")
                            final_fps = float(fps_value)
                            
                    except Exception as e:
                        print(f"[BaBaAi-Tools ERROR] Could not extract valid FPS from value: {repr(fps_value)}. Error: {e}. Using default FPS.")
                        final_fps = default_fps 

                    if final_fps <= 0:
                       print(f"[BaBaAi-Tools WARNING] Extracted FPS ({final_fps}) is not positive. Using default FPS.")
                       final_fps = default_fps

                    video_kwargs["videos_kwargs"]["fps"] = final_fps
                    print(f"[BaBaAi-Tools DEBUG] Final fps value set in video_kwargs: {final_fps}") 
                else:
                     print("[BaBaAi-Tools WARNING] 'fps' key not found in videos_kwargs. Using default FPS.")
                     if "videos_kwargs" not in video_kwargs: video_kwargs["videos_kwargs"] = {}
                     video_kwargs["videos_kwargs"]["fps"] = default_fps 
            else:
                 print("[BaBaAi-Tools DEBUG] 'videos_kwargs' not found or not a dict in video_kwargs. Attempting to set default FPS.")
                 if "videos_kwargs" not in video_kwargs:
                     video_kwargs["videos_kwargs"] = {}
                 elif not isinstance(video_kwargs.get("videos_kwargs"), dict):
                      video_kwargs["videos_kwargs"] = {} 
                 video_kwargs["videos_kwargs"]["fps"] = default_fps

            if 'fps' in video_kwargs:
                print(f"[BaBaAi-Tools DEBUG] Removing redundant top-level 'fps' key from video_kwargs.")
                del video_kwargs['fps']

        inputs = processor(text=[modeltext], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt", **video_kwargs)
        inputs = inputs.to(model.device)
        
        do_sample = temperature > 0.0
        if seed == 0:
             seed = random.randint(0, 0xFFFFFFFFFFFFFFFF)
        torch.manual_seed(seed)

        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        
        if do_sample:
            gen_kwargs.update({"temperature": temperature, "top_p": top_p, "top_k": top_k})

        generated_ids = model.generate(**inputs, **gen_kwargs)

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        text_out = str(output_text[0])
        if "</think>" in text_out:
            text_out = text_out.split("</think>")[-1]
        text_out = re.sub(r"^[\s\u200b\xa0]+", "", text_out)

        return (text_out,)

class Qwen3_Text_Run_Advanced:
    @classmethod
    def INPUT_TYPES(s):
        all_preset_keys = list(Qwen_VL_Run_Advanced._PRESET_PROMPT_MAP.keys())
        text_only_presets = ["无指令"] + [key for key in all_preset_keys if key.startswith("文本指令-")]
        return {"required": {
            "Qwen3_model": ("QWEN3_MODEL",),
            "system_text": ("STRING", {"default": "Text output is limited to 200 words", "multiline": True}),
            "preset_prompt": (text_only_presets, {"default": "无指令"}),
            "text": ("STRING", {"default": "", "multiline": True}),
            "max_new_tokens": ("INT", {"default": 16384, "min": 1, "max": 16384}),
            "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
            "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1}),
            "top_k": ("INT", {"default": 20, "min": 0, "max": 100}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
        }, "optional": {"bypass_model": ("BOOLEAN", {"default": False})}}
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "metrics")
    FUNCTION = "run_model"
    CATEGORY = "📜BaBaAi Tools"

    def run_model(self, Qwen3_model, system_text, preset_prompt, text, max_new_tokens, temperature, top_p, top_k, seed, bypass_model=False):
        if bypass_model:
            return (text, "")
        
        full_text = f"{Qwen_VL_Run_Advanced._PRESET_PROMPT_MAP.get(preset_prompt, '')}\n{text}" if preset_prompt != "无指令" else text
        tokenizer = Qwen3_model["tokenizer"]
        model = Qwen3_model["model"]
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": full_text}
        ]
        model_inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([model_inputs], return_tensors="pt").to(model.device)
        
        do_sample = temperature > 0.0
        if seed == 0:
            seed = random.randint(0, 0xFFFFFFFFFFFFFFFF)
        torch.manual_seed(seed)
        
        streamer = CustomTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = {"do_sample": do_sample, "max_new_tokens": max_new_tokens, "repetition_penalty": 1.2, "no_repeat_ngram_size": 2, "streamer": streamer}
        if do_sample:
            generate_kwargs.update({"temperature": temperature, "top_k": top_k, "top_p": top_p})
            
        try:
            model.generate(**model_inputs, **generate_kwargs)
            output_text = streamer.generated_text
            metrics_text = streamer.get_metrics()
            output_text = re.sub(r'<\|im_end\|>', '', output_text)
        except Exception as e:
            output_text = f"Error during generation: {e}"
            metrics_text = ""
            
        return (output_text, metrics_text)

class Qwen3ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_name": (["unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit", "huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated"], {"default": "huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated"}),
            "quantization": (["none", "4bit", "8bit"], {"default": "4bit"}),
            "attention": (["flash_attention_2", "sdpa", "eager"], {"default": "flash_attention_2"}),
        }}
    RETURN_TYPES = ("QWEN3_MODEL",)
    RETURN_NAMES = ("Qwen3_model",)
    FUNCTION = "load_model"
    CATEGORY = "📜BaBaAi Tools"

    def load_model(self, model_name, quantization, attention):
        model_id = model_name.rsplit("/", 1)[-1]
        model_path = os.path.join(model_directory, model_id)
        if not os.path.exists(model_path):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_name, local_dir=model_path, local_dir_use_symlinks=False)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_compute_dtype=torch.bfloat16, 
                bnb_4bit_use_double_quant=True, 
                llm_int8_enable_fp32_cpu_offload=True,
                bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="balanced", trust_remote_code=True, quantization_config=quant_config, torch_dtype=torch.bfloat16, attn_implementation=attention)
        elif quantization == "8bit":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="balanced", trust_remote_code=True, quantization_config=quant_config, attn_implementation=attention)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="balanced", trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation=attention)
        
        effective_attn = getattr(model.config, "_attn_implementation", "unknown")
        print(f"[BaBaAi-Tools] Model '{model_name}' loaded successfully.")
        print(f"[BaBaAi-Tools] >>> Effective Attention Implementation: {effective_attn.upper()} <<<")
        if attention == "flash_attention_2" and effective_attn != "flash_attention_2":
            print("[BaBaAi-Tools] WARNING: FlashAttention 2 was requested but is not active. The model has fallen back to a different implementation. Check your GPU, CUDA, and flash-attn installation.")
            
        return ({"model": model, "tokenizer": tokenizer},)

NODE_CLASS_MAPPINGS = {
    "Qwen2_5_VLModelLoader": Qwen2_5_VLModelLoader,
    "Qwen3_VL_ModelLoader": Qwen3_VL_ModelLoader,
    "Qwen_VL_Run_Advanced": Qwen_VL_Run_Advanced,
    "Qwen3ModelLoader": Qwen3ModelLoader,
    "Qwen3_Text_Run_Advanced": Qwen3_Text_Run_Advanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen2_5_VLModelLoader": "Qwen2.5-VL模型加载器",
    "Qwen3_VL_ModelLoader": "Qwen3-VL模型加载器",
    "Qwen_VL_Run_Advanced": "Qwen-VL高级运行器",
    "Qwen3ModelLoader": "Qwen3模型加载器",
    "Qwen3_Text_Run_Advanced": "Qwen3文本高级运行器",
}