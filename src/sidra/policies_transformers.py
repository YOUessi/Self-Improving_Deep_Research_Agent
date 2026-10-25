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

    def __init__(self, model: str, max_new_tokens: int = 280):
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
        self.max_new_tokens = max_new_tokens
        # local_files_only prevents implicit network / uncaught model changes.
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, local_files_only=True, dtype="auto", device_map="auto",
        ).eval()

    def decide(self, question: str, observations: list[dict], remaining: int) -> Action:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps({
                "question": question,
                "remaining_steps": remaining,
                "tool_history": observations,
            }, ensure_ascii=False)},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        completion = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        # Never log raw model chain-of-thought, only the validated structured action.
        return Action.from_dict(_json_object(completion))