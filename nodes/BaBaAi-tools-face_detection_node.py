import cv2
import numpy as np
import torch
import logging
import os
from PIL import Image
from typing import Tuple, List, Optional
import folder_paths
import urllib.request

try:
    from comfy_api.v0_0_3_io import (
        ComfyNode, Schema, InputBehavior, NumberDisplay,
        IntegerInput, MaskInput, ImageInput, ImageOutput, ComboInput, CustomInput,
        IntegerOutput, NodeOutput,
    )
    COMFY_V3_AVAILABLE = True
except ImportError:
    ComfyNode = object
    Schema = None
    InputBehavior = None
    NumberDisplay = None
    ImageInput = None
    ImageOutput = None
    ComboInput = None
    CustomInput = None
    IntegerInput = None
    NodeOutput = None
    COMFY_V3_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

log_level = os.getenv('COMFYUI_FACE_DETECTION_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)

if not ULTRALYTICS_AVAILABLE:
    logger.warning("Ultralytics (YOLO) library not found. 'ultralytics' detector type will not be available.")
    logger.warning("Please run: pip install ultralytics")

try:
    models_dir = os.path.abspath(os.path.join(folder_paths.get_folder_paths('checkpoints')[0], ".."))
    ULTRALYTICS_MODEL_DIR = os.path.join(models_dir, "ultralytics", "bbox")
except Exception:
    logger.warning("Could not determine ComfyUI models directory automatically.")
    ULTRALYTICS_MODEL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "models", "ultralytics", "bbox")

os.makedirs(ULTRALYTICS_MODEL_DIR, exist_ok=True)
logger.info(f"YOLO models will be loaded from: {ULTRALYTICS_MODEL_DIR}")

YOLO_MODELS = {}

YOLO_MODEL_LIST = [
    "yolo12n.pt",
    "yolo12s.pt",
    "yolo12m.pt",
    "face_yolov8n.pt",
    "face_yolov8s.pt",
    "face_yolov8m.pt",
    "hand_yolov8n.pt",
    "hand_yolov8s.pt",
    "full_eyes_detect_v1.pt",
    "watermarks_s_yolov8_v1.pt"
]

THIRD_PARTY_YOLO_URLS = {
    "face_yolov8n.pt": "https://huggingface.co/Tenofas/ComfyUI/resolve/main/ultralytics/bbox/face_yolov8n.pt",
    "face_yolov8s.pt": "https://huggingface.co/Tenofas/ComfyUI/resolve/main/ultralytics/bbox/face_yolov8s.pt",
    "face_yolov8m.pt": "https://huggingface.co/Tenofas/ComfyUI/resolve/main/ultralytics/bbox/face_yolov8m.pt",
    "hand_yolov8n.pt": "https://huggingface.co/Tenofas/ComfyUI/resolve/main/ultralytics/bbox/hand_yolov8n.pt",
    "hand_yolov8s.pt": "https://huggingface.co/Tenofas/ComfyUI/resolve/main/ultralytics/bbox/hand_yolov8s.pt",
    "full_eyes_detect_v1.pt": "https://huggingface.co/Tenofas/ComfyUI/resolve/main/ultralytics/bbox/full_eyes_detect_v1.pt",
    "watermarks_s_yolov8_v1.pt": "https://huggingface.co/mnemic/watermarks_yolov8/resolve/main/watermarks_s_yolov8_v1.pt"
}

def get_yolo_model(model_name="yolo12n.pt"):

    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("Ultralytics library is required for YOLO detection.")
    
    model_path = os.path.join(ULTRALYTICS_MODEL_DIR, model_name)

    if model_name not in YOLO_MODELS:
        logger.info(f"Looking for YOLO model at: {model_path}")
        
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}.")
            
            if model_name in THIRD_PARTY_YOLO_URLS:
                url = THIRD_PARTY_YOLO_URLS[model_name]
                logger.warning(f"Attempting to download third-party model from: {url}")
                try:
                    logger.info(f"Downloading {model_name} to {model_path}...")
                    urllib.request.urlretrieve(url, model_path)
                    logger.info(f"Successfully downloaded {model_name}.")
                except Exception as e:
                    logger.error(f"Failed to download third-party model {model_name} from {url}. Please download it manually and place it in {ULTRALYTICS_MODEL_DIR}. Error: {e}")
                    return None
            
            elif model_name.startswith("yolo12"): 
                logger.warning(f"Attempting to auto-download official model {model_name} TO {ULTRALYTICS_MODEL_DIR}...")
                original_cwd = os.getcwd()
                try:
                    os.chdir(ULTRALYTICS_MODEL_DIR)
                    YOLO(model_name)
                    logger.info(f"Successfully auto-downloaded {model_name}.")
                except Exception as e:
                    logger.error(f"Failed to auto-download {model_name}. Please download it manually and place it in {ULTRALYTICS_MODEL_DIR}. Error: {e}")
                    return None
                finally:
                    os.chdir(original_cwd)
            
            else:
                logger.error(f"Model {model_name} is not an official auto-download model and not in the third-party list. Please download it manually and place it in {ULTRALYTICS_MODEL_DIR}.")
                return None
        
        try:
            logger.info(f"Loading YOLO model from: {model_path}")
            YOLO_MODELS[model_name] = YOLO(model_path)
            logger.info(f"Successfully loaded YOLO model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model from path {model_path}: {e}")
            return None

    return YOLO_MODELS[model_name]

def _detect_faces_yolo(image_np: np.ndarray, model_name: str, conf_threshold: float, min_size: int) -> List:

    model = get_yolo_model(model_name)
    if model is None:
        return []

    try:
        results = model(image_np, verbose=False)
        boxes = []
        
        if results and results[0].boxes:
            for box in results[0].boxes:
                conf = box.conf.item()
                
                if conf >= conf_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    w = x2 - x1
                    h = y2 - y1
                    
                    if w >= min_size and h >= min_size:
                        boxes.append((x1, y1, w, h))
                        
        return boxes
        
    except Exception as e:
        logger.error(f"Error during YOLO detection: {e}")
        return []

if COMFY_V3_AVAILABLE:
    class FaceDetectionNode(ComfyNode):
        
        DETECTOR_TYPES = ["opencv_face"]
        if ULTRALYTICS_AVAILABLE:
            DETECTOR_TYPES.append("ultralytics")
        
        YOLO_MODELS_CLASS_VAR = YOLO_MODEL_LIST if ULTRALYTICS_AVAILABLE else []

        @classmethod
        def DEFINE_SCHEMA(cls):
            inputs = [
                ImageInput("image", display_name="Input Image"),
                ComboInput("detector_type", options=cls.DETECTOR_TYPES,
                            default="opencv_face",
                            tooltip="Detection model type: 'opencv_face' (Haar Human Face) or 'ultralytics' (YOLO)"),
                CustomInput("detection_threshold", io_type="FLOAT",
                            min=0.1, max=1.0, default=0.4,
                            tooltip="Confidence threshold. (Mainly used by YOLO, e.g., 0.25-0.8. OpenCV uses minNeighbors)",
                            display_mode=NumberDisplay.slider),
                IntegerInput("min_face_size", display_name="Min Object Size",
                            min=32, max=512, default=32,
                            tooltip="Minimum size for detected objects/faces",
                            display_mode=NumberDisplay.slider),
                IntegerInput("padding", display_name="Padding",
                            min=0, max=512, default=32,
                            tooltip="Padding around detected objects",
                            display_mode=NumberDisplay.slider),
                ComboInput("output_mode", options=["largest_face", "all_faces"],
                            tooltip="Output mode for detected objects"),
                ComboInput("face_output_format", options=["strip", "individual"],
                            tooltip="Format for multiple objects: strip (horizontal layout) or individual (separate batch items). Only applies when output_mode='all_faces'. Max size: 512px.",
                            behavior=InputBehavior.optional),
                ComboInput("ultralytics_model", options=cls.YOLO_MODELS_CLASS_VAR,
                            default=cls.YOLO_MODELS_CLASS_VAR[0] if cls.YOLO_MODELS_CLASS_VAR else None,
                            tooltip="[Ultralytics Only] Choose specific YOLO model (Always visible, but only used if detector_type is 'ultralytics')",
                            behavior=InputBehavior.optional if ULTRALYTICS_AVAILABLE else InputBehavior.disabled),
                ComboInput("classifier_type", options=["default", "alternative"],
                            tooltip="[OpenCV Only] Haar cascade classifier type",
                            behavior=InputBehavior.optional),
            ]
            
            if not ULTRALYTICS_AVAILABLE:
                inputs = [inp for inp in inputs if inp.id_name != "ultralytics_model"]

            return Schema(
                node_id="FaceDetectionNode",
                display_name="Face Detection and Crop",
                description="Detect and crop faces (OpenCV) or all objects (Ultralytics/YOLO).",
                category="📜BaBaAi Tools",
                inputs=inputs,
                outputs=[
                    ImageOutput("cropped_faces", display_name="Cropped Objects",
                              tooltip="Detected and cropped objects/faces"),
                ],
                is_output_node=False,
            )

        @staticmethod
        def _get_cascade_classifiers():

            default_cascade = None
            alternative_cascade = None
            
            try:
                default_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if os.path.exists(default_path):
                    default_cascade = cv2.CascadeClassifier(default_path)
                    if default_cascade.empty():
                        logger.error(f"Failed to load cascade from {default_path}")
                        default_cascade = None
                else:
                    logger.error(f"Default cascade file not found: {default_path}")
                
                alt_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
                if os.path.exists(alt_path):
                    alternative_cascade = cv2.CascadeClassifier(alt_path)
                    if alternative_cascade.empty():
                        logger.warning(f"Failed to load alternative cascade from {alt_path}")
                        alternative_cascade = None
                else:
                    logger.warning(f"Alternative cascade file not found: {alt_path}")
                    
            except Exception as e:
                logger.error(f"Error initializing cascade classifiers: {str(e)}")
                default_cascade = None
                alternative_cascade = None
                
            return default_cascade, alternative_cascade

        @staticmethod
        def add_padding(image: np.ndarray, face_rect: Tuple[int, int, int, int], padding: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:

            x, y, w, h = face_rect
            height, width = image.shape[:2]
            
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(width, x + w + padding)
            y2 = min(height, y + h + padding)
            
            return image[y1:y2, x1:x2], (x1, y1, x2-x1, y2-y1)

        @staticmethod
        def _process_individual_faces(cropped_faces: List[np.ndarray]) -> torch.Tensor:

            if not cropped_faces:
                return torch.zeros((1, 512, 512, 3))

            max_height = min(512, max(face.shape[0] for face in cropped_faces))
            max_width = min(512, max(face.shape[1] for face in cropped_faces))
            
            target_size = (max_width, max_height)
            resized_faces = []
            for face in cropped_faces:
                resized = cv2.resize(face, target_size)
                resized_faces.append(resized)
            
            result_batch = np.stack(resized_faces, axis=0)
            
            if result_batch.shape[3] == 1:
                result_batch = np.repeat(result_batch, 3, axis=3)
            elif result_batch.shape[3] == 4:
                result_batch = result_batch[:, :, :, :3]
            
            result = torch.from_numpy(result_batch).float() / 255.0
            
            assert result.shape[3] == 3, f"Output must have 3 channels, got {result.shape[3]}"
            
            return result

        @classmethod
        async def execute(cls, image: torch.Tensor, detector_type: str, detection_threshold: float, min_face_size: int, 
                         padding: int, output_mode: str, face_output_format: str = "strip",
                         classifier_type: str = "default", ultralytics_model: str = None, 
                         mask: torch.Tensor = None) -> NodeOutput:
            
            if ultralytics_model is None and detector_type == "ultralytics" and cls.YOLO_MODELS_CLASS_VAR:
                ultralytics_model = cls.YOLO_MODELS_CLASS_VAR[0]

            if isinstance(image, torch.Tensor):
                logger.debug(f"Processing tensor - Shape: {image.shape}, Type: {image.dtype}")
                
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                elif len(image.shape) != 4:
                    raise ValueError(f"Expected 3D or 4D tensor, got shape: {image.shape}")
                
                B, H, W, C = image.shape

                if C == 1:
                    image = image.repeat(1, 1, 1, 3)
                elif C == 4:
                    image = image[:, :, :, :3]
                elif C > 4:
                    logger.warning(f"Input has {C} channels, using first 3")
                    image = image[:, :, :, :3]
                elif C != 3:
                    raise ValueError(f"Cannot handle {C} channels")
                
                image_np = image[0].cpu().numpy()
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
                    
            else:
                image_np = image

            if not isinstance(image_np, np.ndarray) or len(image_np.shape) != 3:
                raise ValueError(f"Expected 3D numpy array, got {type(image_np)} with shape {getattr(image_np, 'shape', 'unknown')}")
            
            if image_np.shape[2] != 3:
                raise ValueError(f"Expected RGB image (3 channels), got {image_np.shape[2]} channels")

            faces = []
            
            if detector_type == "opencv_face":
                logger.info("Using OpenCV Haar cascade detector.")
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                
                default_cascade, alternative_cascade = cls._get_cascade_classifiers()
                
                if classifier_type == "alternative":
                    if alternative_cascade is None:
                        logger.warning("Alternative Haar cascade not available, falling back to default")
                        if default_cascade is None:
                            logger.error("No cascade classifiers available")
                            return NodeOutput(cropped_faces=torch.zeros((1, 512, 512, 3)))
                        face_cascade = default_cascade
                    else:
                        face_cascade = alternative_cascade
                else:
                    if default_cascade is None:
                        logger.error("Default Haar cascade not available")
                        return NodeOutput(cropped_faces=torch.zeros((1, 512, 512, 3)))
                    face_cascade = default_cascade
                
                try:
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(min_face_size, min_face_size)
                    )
                except Exception as e:
                    logger.error(f"Haar face detection failed: {str(e)}")
                    faces = []
            
            elif detector_type == "ultralytics" and ULTRALYTICS_AVAILABLE:
                if not ultralytics_model:
                     logger.error("Ultralytics detector selected, but no model was specified.")
                     return NodeOutput(cropped_faces=torch.zeros((1, 512, 512, 3)))
                
                logger.info(f"Using Ultralytics (YOLO) detector with model: {ultralytics_model}")
                model_name = ultralytics_model
                faces = _detect_faces_yolo(image_np, model_name, detection_threshold, min_face_size)
            
            else:
                if not ULTRALYTICS_AVAILABLE and detector_type == "ultralytics":
                    logger.error("YOLO detection failed. Ultralytics library not found. Please run 'pip install ultralytics'")
                else:
                    logger.error(f"Unknown or unsupported detector_type: {detector_type}")
                faces = []

            if len(faces) == 0:
                logger.warning(f"No objects/faces detected with detector '{detector_type}'")
                return NodeOutput(cropped_faces=torch.zeros((1, 512, 512, 3)))

            cropped_faces = []
            for x, y, w, h in faces:
                face_img, _ = cls.add_padding(image_np, (x, y, w, h), padding)
                cropped_faces.append(face_img)

            if output_mode == "largest_face":
                largest_face = max(cropped_faces, key=lambda x: x.shape[0] * x.shape[1])
                cropped_faces = [largest_face]

            if output_mode == "all_faces" and len(cropped_faces) > 1 and face_output_format == "individual":
                result = cls._process_individual_faces(cropped_faces)
                return NodeOutput(cropped_faces=result)
                
            elif len(cropped_faces) > 1:
                max_height = min(512, max(face.shape[0] for face in cropped_faces))
                resized_faces = []
                for face in cropped_faces:
                    aspect_ratio = face.shape[1] / face.shape[0]
                    new_width = int(max_height * aspect_ratio)
                    resized = cv2.resize(face, (new_width, max_height))
                    resized_faces.append(resized)
                result = np.hstack(resized_faces)
            else:
                result = cropped_faces[0]
            
            if result.shape[2] == 1:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
            elif result.shape[2] == 4:
                result = cv2.cvtColor(result, cv2.COLOR_RGBA2RGB)
            
            result = torch.from_numpy(result).float() / 255.0
            result = result.unsqueeze(0)
            
            assert result.shape[3] == 3, f"Output must have 3 channels, got {result.shape[3]}"
            
            return NodeOutput(cropped_faces=result)

        @classmethod
        def IS_CHANGED(cls, **kwargs):
            return ""

class FaceDetectionNodeV1:

    DETECTOR_TYPES = ["opencv_face"]
    if ULTRALYTICS_AVAILABLE:
        DETECTOR_TYPES.append("ultralytics")

    YOLO_MODELS_CLASS_VAR = YOLO_MODEL_LIST if ULTRALYTICS_AVAILABLE else []

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "detector_type": (s.DETECTOR_TYPES, {"default": "opencv_face"}),
                "detection_threshold": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Confidence threshold. (Mainly used by YOLO, e.g., 0.25-0.8. OpenCV uses minNeighbors)"
                }),
                "min_face_size": ("INT", {
                    "default": 32,
                    "min": 32,
                    "max": 512,
                    "step": 8,
                    "display_name": "Min Object Size"
                }),
                "padding": ("INT", {
                    "default": 32,
                    "min": 0,
                    "max": 512,
                    "step": 8
                }),
                "output_mode": (["largest_face", "all_faces"],),
            },
            "optional": {
                "face_output_format": (["strip", "individual"], {"default": "strip", "tooltip": "Format for multiple objects: strip (horizontal layout) or individual (separate batch items). Only applies when output_mode='all_faces'. Max size: 512px."}),
                "classifier_type": (["default", "alternative"], {"default": "default", "tooltip": "[OpenCV Only] Haar cascade classifier type"}),
            }
        }

        if ULTRALYTICS_AVAILABLE:
            inputs["optional"]["ultralytics_model"] = (s.YOLO_MODELS_CLASS_VAR, {
                "default": s.YOLO_MODELS_CLASS_VAR[0] if s.YOLO_MODELS_CLASS_VAR else "yolo12n.pt",
                "tooltip": "[Ultralytics Only] Choose specific YOLO model (Always visible, but only used if detector_type is 'ultralytics')"
            })
        
        return inputs


    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Cropped Objects",)
    FUNCTION = "detect_and_crop_faces"
    CATEGORY = "📜BaBaAi Tools"

    def __init__(self):
        self.default_cascade = None
        self.alternative_cascade = None
        
        try:
            default_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(default_path):
                self.default_cascade = cv2.CascadeClassifier(default_path)
                if self.default_cascade.empty():
                    logger.error(f"Failed to load cascade from {default_path}")
                    self.default_cascade = None
            else:
                logger.error(f"Default cascade file not found: {default_path}")
            
            alt_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
            if os.path.exists(alt_path):
                self.alternative_cascade = cv2.CascadeClassifier(alt_path)
                if self.alternative_cascade.empty():
                    logger.warning(f"Failed to load alternative cascade from {alt_path}")
                    self.alternative_cascade = None
            else:
                logger.warning(f"Alternative cascade file not found: {alt_path}")
                
        except Exception as e:
            logger.error(f"Error initializing cascade classifiers: {str(e)}")
            self.default_cascade = None
            self.alternative_cascade = None

    def add_padding(self, image: np.ndarray, face_rect: Tuple[int, int, int, int], padding: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:

        x, y, w, h = face_rect
        height, width = image.shape[:2]
        
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)
        
        return image[y1:y2, x1:x2], (x1, y1, x2-x1, y2-y1)

    def _process_individual_faces(self, cropped_faces: List[np.ndarray]) -> torch.Tensor:

        if not cropped_faces:
            return torch.zeros((1, 512, 512, 3))
            
        if COMFY_V3_AVAILABLE:
            return FaceDetectionNode._process_individual_faces(cropped_faces)
        else:
            max_height = min(512, max(face.shape[0] for face in cropped_faces))
            max_width = min(512, max(face.shape[1] for face in cropped_faces))
            
            target_size = (max_width, max_height)
            resized_faces = []
            for face in cropped_faces:
                resized = cv2.resize(face, target_size)
                resized_faces.append(resized)
            
            result_batch = np.stack(resized_faces, axis=0)
            
            if result_batch.shape[3] == 1:
                result_batch = np.repeat(result_batch, 3, axis=3)
            elif result_batch.shape[3] == 4:
                result_batch = result_batch[:, :, :, :3]
            
            result = torch.from_numpy(result_batch).float() / 255.0
            assert result.shape[3] == 3, f"Output must have 3 channels, got {result.shape[3]}"
            
            return result

    def detect_and_crop_faces(self, image, detector_type, detection_threshold, min_face_size, padding, output_mode, face_output_format="strip", classifier_type="default", ultralytics_model=None):
        
        if ultralytics_model is None and detector_type == "ultralytics" and self.YOLO_MODELS_CLASS_VAR:
            ultralytics_model = self.YOLO_MODELS_CLASS_VAR[0]

        if isinstance(image, torch.Tensor):
            logger.debug(f"Processing tensor - Shape: {image.shape}, Type: {image.dtype}")
            
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            elif len(image.shape) != 4:
                raise ValueError(f"Expected 3D or 4D tensor, got shape: {image.shape}")
            
            B, H, W, C = image.shape
            
            if C == 1:
                image = image.repeat(1, 1, 1, 3)
            elif C == 4:
                image = image[:, :, :, :3]
            elif C > 4:
                logger.warning(f"Input has {C} channels, using first 3")
                image = image[:, :, :, :3]
            elif C != 3:
                raise ValueError(f"Cannot handle {C} channels")
            
            image_np = image[0].cpu().numpy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = np.clip(image_np, 0, 255).astype(np.uint8)
                
        else:
            image_np = image

        if not isinstance(image_np, np.ndarray) or len(image_np.shape) != 3:
            raise ValueError(f"Expected 3D numpy array, got {type(image_np)} with shape {getattr(image_np, 'shape', 'unknown')}")
        
        if image_np.shape[2] != 3:
            raise ValueError(f"Expected RGB image (3 channels), got {image_np.shape[2]} channels")

        faces = []
        
        if detector_type == "opencv_face":
            logger.info("Using OpenCV Haar cascade detector.")
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

            if classifier_type == "alternative":
                if self.alternative_cascade is None:
                    logger.warning("Alternative Haar cascade not available, falling back to default")
                    if self.default_cascade is None:
                        logger.error("No cascade classifiers available")
                        return (torch.zeros((1, 512, 512, 3)),)
                    face_cascade = self.default_cascade
                else:
                    face_cascade = self.alternative_cascade
            else:
                if self.default_cascade is None:
                    logger.error("Default Haar cascade not available")
                    return (torch.zeros((1, 512, 512, 3)),)
                face_cascade = self.default_cascade
            
            try:
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(min_face_size, min_face_size)
                )
            except Exception as e:
                logger.error(f"Haar face detection failed: {str(e)}")
                faces = []

        elif detector_type == "ultralytics" and ULTRALYTICS_AVAILABLE:
            if not ultralytics_model:
                     logger.error("Ultralytics detector selected, but no model was specified.")
                     return (torch.zeros((1, 512, 512, 3)),)

            logger.info(f"Using Ultralytics (YOLO) detector with model: {ultralytics_model}")
            model_name = ultralytics_model
            faces = _detect_faces_yolo(image_np, model_name, detection_threshold, min_face_size)
        
        else:
            if not ULTRALYTICS_AVAILABLE and detector_type == "ultralytics":
                logger.error("YOLO detection failed. Ultralytics library not found. Please run 'pip install ultralytics'")
            else:
                logger.error(f"Unknown or unsupported detector_type: {detector_type}")
            faces = []


        if len(faces) == 0:
            logger.warning(f"No objects/faces detected with detector '{detector_type}'")
            return (torch.zeros((1, 512, 512, 3)),)

        cropped_faces = []
        for x, y, w, h in faces:
            face_img, _ = self.add_padding(image_np, (x, y, w, h), padding)
            cropped_faces.append(face_img)

        if output_mode == "largest_face":
            largest_face = max(cropped_faces, key=lambda x: x.shape[0] * x.shape[1])
            cropped_faces = [largest_face]

        if output_mode == "all_faces" and len(cropped_faces) > 1 and face_output_format == "individual":
            result = self._process_individual_faces(cropped_faces)
            return (result,)
            
        elif len(cropped_faces) > 1:
            max_height = min(512, max(face.shape[0] for face in cropped_faces))
            resized_faces = []
            for face in cropped_faces:
                aspect_ratio = face.shape[1] / face.shape[0]
                new_width = int(max_height * aspect_ratio)
                resized = cv2.resize(face, (new_width, max_height))
                resized_faces.append(resized)
            result = np.hstack(resized_faces)
        else:
            result = cropped_faces[0]
        
        if result.shape[2] == 1:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        elif result.shape[2] == 4:
            result = cv2.cvtColor(result, cv2.COLOR_RGBA2RGB)
        
        result = torch.from_numpy(result).float() / 255.0
        result = result.unsqueeze(0)
        
        assert result.shape[3] == 3, f"Output must have 3 channels, got {result.shape[3]}"
        
        return (result,)

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return ""

if COMFY_V3_AVAILABLE:
    NODE_CLASS_MAPPINGS = {
        "FaceDetectionNode": FaceDetectionNode
    }
else:
    NODE_CLASS_MAPPINGS = {
        "FaceDetectionNode": FaceDetectionNodeV1
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDetectionNode": "物体检测和裁剪"
}