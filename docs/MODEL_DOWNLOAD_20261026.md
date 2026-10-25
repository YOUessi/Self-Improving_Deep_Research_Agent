# Tang model downloads: Qwen3-8B + Qwen3-8B-AWQ (2026-10-26)

## Why Hugging Face was unreachable

HTTPS + DNS probes on Tang and Mac:
- Tang `huggingface.co` unexpectedly resolved to `157.240.16.50`; TLS connection to port 443 timed out.
- Mac independently resolved `huggingface.co` to `108.160.165.173`; connection timed out.
- `https://modelscope.cn/api/v1/models/Qwen/Qwen3-8B`, `https://hf-mirror.com/Qwen/Qwen3-8B/resolve/main/config.json`, and GitHub were HTTP 200 from Tang.
- Evidence is consistent with local resolver or network-routing interference; this does not establish the exact root cause or justify changing global DNS settings.

GitHub's [Qwen3 code repository](https://github.com/QwenLM/Qwen3) does not directly contain the large checkpoint weights. The Qwen project itself recommends [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-8B) when Hugging Face is unavailable. Git LFS for `https://huggingface.co/Qwen/... ` would still rely on the inaccessible model host.

## Download configuration

Isolated downloader: `/home/you/experiments/sidra_download/.venv`, `modelscope==1.40.2`. Neither system Python nor the CodeRL-Lab venv was modified.

| Model | Official ModelScope ID | Tang local directory | Expected tensor shards |
| --- | --- | --- | --- |
| Qwen3-8B BF16 | Qwen/Qwen3-8B | /home/you/models/Qwen3-8B | 5 |
| Qwen3-8B AWQ | Qwen/Qwen3-8B-AWQ | /home/you/models/Qwen3-8B-AWQ | 2 |

Commands:

```bash
/home/you/experiments/sidra_download/.venv/bin/modelscope download Qwen/Qwen3-8B --revision master --max-workers 3 --local-dir /home/you/models/Qwen3-8B
/home/you/experiments/sidra_download/.venv/bin/modelscope download Qwen/Qwen3-8B-AWQ --revision master --max-workers 3 --local-dir /home/you/models/Qwen3-8B-AWQ
```

Downloads began 2026-10-26 through ModelScope and write resumable partial files. Logs are at:
- `/home/you/experiments/sidra_download/logs/Qwen3-8B.log`
- `/home/you/experiments/sidra_download/logs/Qwen3-8B-AWQ.log`

No checkpoint binaries are stored in GitHub.

## Verification gate

The official ModelScope file listing includes expected bytes and SHA-256 for all `.safetensors` shards and tokenizer/index files. Values are pinned in [verify_modelscope_models.py](../scripts/verify_modelscope_models.py).

```bash
python /home/you/experiments/sidra_download/verify_modelscope_models.py --base-dir /home/you/models
```

Exit code **0** with `ready=true` for both is required before saying either model is downloaded completely. Checking directory existence or config.json alone is insufficient. The ModelScope download process must also exit successfully and leave no `.incomplete` files.

**Resource note:** full BF16 8B weights alone occupy roughly 16.4 GB (decimal), so one 16GB GPU is not sufficient for comfortable full-precision inference with KV cache; AWQ 4-bit is the preferred inference candidate after an independent runtime compatibility test. Merely passing SHA verification is not proof that AWQ is compatible with the installed Transformers/vLLM stack.

Status at initialization: **downloads started, final checksum verification pending**. No model inference or training result was claimed.
