"""GPU inference policy for cached local Hugging Face instruct models.

This adapter runs *pretrained* model inference only. It does not train a model.
No external server, API key or internet access is needed if weights are cached.
"""

from __future__ import annotations

import json

from sidra.policies import SYSTEM, _json_object
from sidra.schema import Action


class TransformersPolicy:
    name = "transformers_pretrained_agent"

    def __init__(self, model: str, max_new_tokens: int = 280,
                 offload_gpu_gib: int = 0, offload_dir: str | None = None,
                 adapter_path: str | None = None):
        try:
            # Tang has Pillow 9.0.1. Transformers 4.57 image_utils references
            # PIL.Image.Resampling which was introduced in newer Pillow.
            # Provide a *process-local* API alias, without changing system packages.
            try:
                from PIL import Image
                if not hasattr(Image, "Resampling"):
                    Image.Resampling = Image
            except ImportError:
                pass
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Requires torch + transformers (run on Tang)") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable: use GPU Tang or the HTTP endpoint policy")
        self.torch = torch
        self.model_name = model
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.model_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        # Offload is optional for full 8B BF16 on a 16GB GPU. The AWQ/vLLM
        # serving path remains the preferred high-throughput option.
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        opts = {}
        if offload_gpu_gib:
            if offload_gpu_gib < 1:
                raise ValueError("offload_gpu_gib must be positive")
            opts = {"device_map": "auto",
                    "max_memory": {0: f"{offload_gpu_gib}GiB", "cpu": "22GiB"}}
            if offload_dir:
                opts["offload_folder"] = offload_dir
            self.model = AutoModelForCausalLM.from_pretrained(
                model, local_files_only=True, dtype="auto", **opts).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model, local_files_only=True, dtype="auto").to("cuda").eval()
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("PEFT is required to evaluate an SFT adapter") from exc
            from pathlib import Path
            adapter_root = Path(adapter_path)
            if not (adapter_root / "adapter_config.json").is_file():
                raise ValueError("Missing adapter_config.json for SFT checkpoint")
            if not ((adapter_root / "adapter_model.safetensors").is_file()
                    or (adapter_root / "adapter_model.bin").is_file()):
                raise ValueError("Missing LoRA adapter weights")
            self.model = PeftModel.from_pretrained(
                self.model, str(adapter_root), local_files_only=True,
                is_trainable=False).eval()

    def decide(self, question: str, observations: list[dict], remaining: int) -> Action:
        return self.decide_with_system(question, observations, remaining, SYSTEM)

    def decide_with_system(self, question: str, observations: list[dict],
                           remaining: int, system_prompt: str) -> Action:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps({
                "question": question,
                "remaining_steps": remaining,
                "tool_history": observations,
            }, ensure_ascii=False)},
        ]
        # Qwen3 has distinct thinking/non-thinking chat templates. For
        # structured JSON tool actions use its non-thinking mode first.
        # Do not silently apply this option to older model families.
        template_options = {"enable_thinking": False} if "qwen3" in self.model_name.casefold() else {}
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_options
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        self.model_calls += 1
        self.prompt_tokens += int(inputs["input_ids"].shape[1])
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        self.completion_tokens += int(output_ids.shape[-1] - inputs["input_ids"].shape[1])
        completion = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        # Never log raw model chain-of-thought, only the validated structured action.
        return Action.from_dict(_json_object(completion))