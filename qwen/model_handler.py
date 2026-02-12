"""
Model handler for Qwen3-VL models
"""
import time
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import torch.nn.functional as F
from dataclasses import dataclass
from PIL import Image
import torch

logger = logging.getLogger(__name__)


def _summarize_hf_device_map(device_map: Any) -> Dict[str, int]:
    """Summarize an Accelerate/HF device map.

    Returns counts of modules assigned per device string (e.g., "cuda:0", "cpu", "disk").
    """
    summary: Dict[str, int] = {}
    if not isinstance(device_map, dict):
        return summary

    cuda_available = torch.cuda.is_available()

    def _normalize_device(dev: Any) -> str:
        if isinstance(dev, torch.device):
            return str(dev)
        if isinstance(dev, int):
            return f"cuda:{dev}" if cuda_available else str(dev)
        dev_str = str(dev)
        if cuda_available and dev_str.isdigit():
            return f"cuda:{dev_str}"
        # Common accelerate labels: 'cpu', 'disk', 'cuda:0'
        return dev_str

    for _, dev in device_map.items():
        dev_str = _normalize_device(dev)
        summary[dev_str] = summary.get(dev_str, 0) + 1
    return summary


def _log_runtime_device_info(model: Any) -> None:
    """Log information that helps confirm GPU usage and detect offloading."""
    try:
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_available}")
        if cuda_available:
            try:
                device_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                total_gb = float(props.total_memory) / (1024 ** 3)
                logger.info(f"CUDA device 0: {device_name} ({total_gb:.1f} GiB)")
                logger.info(
                    "CUDA mem (GiB): "
                    f"allocated={torch.cuda.memory_allocated(0)/(1024**3):.2f}, "
                    f"reserved={torch.cuda.memory_reserved(0)/(1024**3):.2f}"
                )
            except Exception as e:
                logger.info(f"CUDA device query failed (non-fatal): {e}")

        # Hugging Face / Accelerate device placement summary
        device_map = getattr(model, "hf_device_map", None) or getattr(model, "_hf_device_map", None)
        if isinstance(device_map, dict):
            summary = _summarize_hf_device_map(device_map)
            logger.info(f"hf_device_map summary (module shards per device): {summary}")

            offload_targets = {k for k in summary.keys() if k.startswith("cpu") or k.startswith("disk")}
            if offload_targets:
                logger.warning(
                    "Model appears partially offloaded to non-GPU devices: "
                    f"{sorted(offload_targets)}. This is expected if GPU VRAM is insufficient and can slow inference."
                )
        else:
            # Fallback: best-effort device from first parameter
            try:
                first_param_device = next(model.parameters()).device
                logger.info(f"Model parameter device (first param): {first_param_device}")
            except Exception:
                pass
    except Exception as e:
        logger.info(f"Device info logging failed (non-fatal): {e}")


@dataclass
class InferenceResult:
    """Result from a single model inference"""
    output_text: str
    inference_time_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    raw_input: Optional[Dict[str, Any]] = None
    # Added for confidence extraction (Confidence Under the Hood paper)
    logits: Optional[torch.Tensor] = None  # All generated token logits stacked [seq_len, vocab_size]
    token_probabilities: Optional[Dict[int, float]] = None  # Token ID -> probability (first token only, for backwards compat)
    generated_token_ids: Optional[List[int]] = None  # All generated token IDs


class Qwen3VLModel:
    """Handler for Qwen3-VL models"""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        use_flash_attention: bool = True,
        use_torch_compile: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        do_sample: bool = False
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.use_flash_attention = use_flash_attention
        self.use_torch_compile = use_torch_compile
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        self.model = None
        self.processor = None
        self._loaded = False
    
    def load(self) -> None:
        """Load model and processor"""
        if self._loaded:
            return

        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        logger.info(f"Loading model: {self.model_name}")

        trust_remote_code = False
        if "30b" in self.model_name.lower() or "a3b" in self.model_name.lower():
            logger.info(f"Enabling trust_remote_code=True for {self.model_name}")
            trust_remote_code = True

        # Use dtype="auto" for automatic dtype selection as recommended
        try:
            if self.use_flash_attention:
                # Try with flash attention
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    dtype=self.torch_dtype,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                    trust_remote_code=trust_remote_code
                )
            else:
                # Use default attention (SDPA)
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    dtype=self.torch_dtype,
                    device_map="auto",
                    trust_remote_code=trust_remote_code
                )
        except Exception as e:
            logger.warning(f"Failed to load with specified settings, trying with dtype='auto': {e}")
            # Fallback: use dtype="auto" as in the example
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                dtype="auto",
                device_map="auto",
                trust_remote_code=trust_remote_code
            )

        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=trust_remote_code)

        # Log device placement and potential offloading (cpu/disk) for verification.
        _log_runtime_device_info(self.model)

        # Apply torch.compile if enabled
        if self.use_torch_compile:
            logger.info("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
            logger.info("Model compilation complete")

        self._loaded = True
        logger.info(f"Model loaded successfully: {self.model_name}")
    
    def unload(self) -> None:
        """Unload model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._loaded = False
        logger.info(f"Model unloaded: {self.model_name}")
    
    def generate_text_only(
        self,
        prompt: str,
        return_logits: bool = False
    ) -> InferenceResult:
        """
        Generate response for text-only input (JSON or Markdown tables)
        """
        if not self._loaded:
            self.load()
        
        # Prepare messages for text-only input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template and tokenize
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            truncation=False,
            max_length=None
        )
        inputs = inputs.to(self.model.device)
        
        # Store raw input for logging
        raw_input = {
            "type": "text_only",
            "prompt": prompt,
            "messages": messages
        }
        
        # Generate
        start_time = time.perf_counter()

        # For confidence extraction, we need to get logits
        all_token_logits = None
        token_probs = None

        with torch.no_grad():
            if return_logits:
                # Generate with output_scores to get logits for confidence extraction
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.do_sample else None,
                    do_sample=self.do_sample,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                generated_ids = outputs.sequences

                # Stack all token logits for confidence calculation [seq_len, vocab_size]
                if outputs.scores:
                    # Move to CPU and cast to float32 immediately to prevent VRAM OOM
                    all_token_logits = torch.stack(
                        [s[0].float().cpu() for s in outputs.scores], 
                        dim=0
                    )
                    
                    # Also compute first-token probabilities for backwards compatibility
                    probs = F.softmax(outputs.scores[0][0], dim=-1)
                    top_k = 100
                    top_probs, top_indices = torch.topk(probs, top_k)
                    token_probs = {
                        int(idx): float(prob)
                        for idx, prob in zip(top_indices.tolist(), top_probs.tolist())
                    }
            else:
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.do_sample else None,
                    do_sample=self.do_sample,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )

        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Safe extraction of tokens list
        gen_tokens_list = generated_ids_trimmed[0].tolist() if generated_ids_trimmed else []

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return InferenceResult(
            output_text=output_text.strip(),
            inference_time_ms=inference_time_ms,
            input_tokens=inputs.input_ids.shape[1],
            output_tokens=len(gen_tokens_list),
            raw_input=raw_input,
            logits=all_token_logits,
            token_probabilities=token_probs,
            generated_token_ids=gen_tokens_list
        )

    def generate_with_image(
        self,
        prompt: str,
        image: Image.Image,
        example_image: Optional[Image.Image] = None,
        return_logits: bool = False
    ) -> InferenceResult:
        """
        Generate response for image + text input.
        
        Args:
            prompt: The text prompt
            image: The main table image
            example_image: Optional example image for few-shot prompting
            return_logits: Whether to return logits for confidence extraction
        
        Returns:
            InferenceResult with output and optional logits
        """
        if not self._loaded:
            self.load()
        
        # Prepare messages with image + text (following Qwen3-VL example)
        # For few-shot with example image, include it before the main prompt
        content = []
        if example_image is not None:
            # For few-shot with images, the prompt has <IMAGE> markers where images should go
            if "<IMAGE>" in prompt:
                # Split prompt at <IMAGE> markers and insert images
                parts = prompt.split("<IMAGE>")
                if len(parts) == 3:  # Should have: [before_ex, middle, after_main]
                    # Structure: text -> example_image -> text -> main_image -> text
                    content.append({"type": "text", "text": parts[0]})
                    content.append({"type": "image", "image": example_image})
                    content.append({"type": "text", "text": parts[1]})
                    content.append({"type": "image", "image": image})
                    content.append({"type": "text", "text": parts[2]})
                    logger.info(f"Multi-image message structure: [text, example_image({example_image.size}), text, main_image({image.size}), text]")
                else:
                    # Unexpected format, fallback
                    logger.warning(f"Unexpected <IMAGE> marker count: {len(parts)-1}")
                    content.append({"type": "image", "image": example_image})
                    content.append({"type": "image", "image": image})
                    content.append({"type": "text", "text": prompt.replace("<IMAGE>", "")})
                    logger.info(f"Multi-image fallback: [example_image({example_image.size}), main_image({image.size}), text]")
            else:
                # Old format with "Now analyze this table:" split (for text few-shot)
                if "Now analyze this table:" in prompt:
                    parts = prompt.split("Now analyze this table:")
                    example_part = parts[0]
                    main_part = "Now analyze this table:" + parts[1]
                    
                    content.append({"type": "image", "image": example_image})
                    content.append({"type": "text", "text": example_part})
                    content.append({"type": "image", "image": image})
                    content.append({"type": "text", "text": main_part})
                    logger.info(f"Text-based few-shot with images: [example_image({example_image.size}), text, main_image({image.size}), text]")
                else:
                    # Fallback: just put example image first, then main image with full prompt
                    content.append({"type": "image", "image": example_image})
                    content.append({"type": "image", "image": image})
                    content.append({"type": "text", "text": prompt})
                    logger.info(f"Few-shot fallback: [example_image({example_image.size}), main_image({image.size}), text]")
        else:
            # Single image case (zero-shot or CoT)
            content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": prompt})
            logger.debug(f"Single image: size={image.size}")
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        # Apply chat template and tokenize with image
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            truncation=False,
            max_length=None
        )
        inputs = inputs.to(self.model.device)
        
        # Store raw input for logging (without the actual image data)
        raw_input = {
            "type": "image_text",
            "prompt": prompt,
            "image_size": image.size,
            "image_mode": image.mode
        }
        
        # Generate
        start_time = time.perf_counter()

        # For confidence extraction, we need to get logits
        all_token_logits = None
        token_probs = None

        with torch.no_grad():
            if return_logits:
                # Generate with output_scores to get logits for confidence extraction
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.do_sample else None,
                    do_sample=self.do_sample,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                generated_ids = outputs.sequences

                # Stack all token logits for confidence calculation [seq_len, vocab_size]
                if outputs.scores:
                    # Move to CPU and cast to float32 immediately to prevent VRAM OOM
                    all_token_logits = torch.stack(
                        [s[0].float().cpu() for s in outputs.scores], 
                        dim=0
                    )
                    
                    # Also compute first-token probabilities for backwards compatibility
                    probs = F.softmax(outputs.scores[0][0], dim=-1)
                    top_k = 100
                    top_probs, top_indices = torch.topk(probs, top_k)
                    token_probs = {
                        int(idx): float(prob)
                        for idx, prob in zip(top_indices.tolist(), top_probs.tolist())
                    }
            else:
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.do_sample else None,
                    do_sample=self.do_sample,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )

        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        gen_tokens_list = generated_ids_trimmed[0].tolist() if generated_ids_trimmed else []

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return InferenceResult(
            output_text=output_text.strip(),
            inference_time_ms=inference_time_ms,
            input_tokens=inputs.input_ids.shape[1],
            output_tokens=len(gen_tokens_list),
            raw_input=raw_input,
            logits=all_token_logits,
            token_probabilities=token_probs,
            generated_token_ids=gen_tokens_list
        )

    def generate(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        return_logits: bool = False
    ) -> InferenceResult:
        """
        Unified generate method that handles both text-only and image+text inputs

        Args:
            prompt: The text prompt
            image: Optional PIL Image for multimodal input
            return_logits: If True, return token logits for confidence extraction

        Returns:
            InferenceResult with output text and optionally logits/probabilities
        """
        if image is not None:
            # FIX: Explicitly pass kwargs to avoid passing return_logits into example_image
            return self.generate_with_image(
                prompt=prompt, 
                image=image, 
                example_image=None, 
                return_logits=return_logits
            )
        else:
            return self.generate_text_only(prompt, return_logits)

class ModelManager:
    """Manager for loading/unloading multiple models efficiently"""
    
    def __init__(self, config: Any):
        self.config = config
        self.current_model: Optional[Qwen3VLModel] = None
        self.current_model_name: Optional[str] = None
    
    def get_model(self, model_name: str) -> Qwen3VLModel:
        """
        Get a model, loading it if necessary and unloading the previous one.
        """
        if self.current_model_name == model_name and self.current_model is not None:
            return self.current_model
        
        # Unload current model if any
        if self.current_model is not None:
            logger.info(f"Unloading model: {self.current_model_name}")
            self.current_model.unload()
        
        # Load new model
        logger.info(f"Loading model: {model_name}")
        self.current_model = Qwen3VLModel(
            model_name=model_name,
            device=self.config.device,
            torch_dtype=self.config.torch_dtype,
            use_flash_attention=self.config.use_flash_attention,
            use_torch_compile=self.config.use_torch_compile,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.do_sample
        )
        self.current_model.load()
        self.current_model_name = model_name
        
        return self.current_model
    
    def cleanup(self):
        """Cleanup all resources"""
        if self.current_model is not None:
            self.current_model.unload()
            self.current_model = None
            self.current_model_name = None