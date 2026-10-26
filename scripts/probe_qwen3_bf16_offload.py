"""Small Qwen3 BF16 inference diagnostic with explicit CPU/GPU memory caps.

This is a hardware/model loading smoke test only, not a Search Agent benchmark.
Run with the pre-existing CodeRL-Lab Python environment on Tang. No network
downloads or model training occur.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def run(model_dir: Path, output: Path, *, gpu_gib: int, max_new_tokens: int) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("requires a CUDA GPU")
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    begin = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, device_map="auto",
        local_files_only=True,
        max_memory={0: f"{gpu_gib}GiB", "cpu": "22GiB"},
        offload_folder=str(output.parent / "offload"),
    ).eval()
    load_seconds = round(time.monotonic() - begin, 2)
    message = [{"role": "user", "content": (
        "Return exactly one JSON object with keys kind and query. "
        "Kind must be search, query must be 'Tokyo Tower'."
    )}]
    prompt = tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    first = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(first)
    torch.cuda.reset_peak_memory_stats()
    begin = time.monotonic()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id)
    infer_seconds = round(time.monotonic() - begin, 2)
    answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:],
                              skip_special_tokens=True)
    result = {
        "model_dir": str(model_dir), "mode": "BF16 GPU+CPU offload smoke only",
        "load_seconds": load_seconds, "inference_seconds": infer_seconds,
        "max_new_tokens": max_new_tokens, "generated_tokens": int(
            outputs.shape[-1] - inputs["input_ids"].shape[-1]),
        "max_gpu_memory_gib": gpu_gib,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "device_map": {k: str(v) for k, v in model.hf_device_map.items()},
        "generated_text": answer, "model_trained": False,
        "benchmark_accuracy_claim": False,
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu-gib", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    args = ap.parse_args()
    print(json.dumps(run(args.model, args.out, gpu_gib=args.gpu_gib,
                         max_new_tokens=args.max_new_tokens), indent=2))


if __name__ == "__main__":
    main()
