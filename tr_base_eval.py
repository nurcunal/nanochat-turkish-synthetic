from contextlib import nullcontext

import torch

from nanochat.common import compute_init, compute_cleanup, print0
from nanochat.checkpoint_manager import load_model
from scripts.chat_eval import run_categorical_eval

from tasks.xnli_tr import XNLI_TR
from tasks.xcopa_tr import XCOPA_TR
from tasks.belebele_tr import Belebele_TR


MODEL_TAG = "d4"
MODEL_STEP = 1400

BATCH_SIZE = 8
MAX_PROBLEMS = None  # set e.g. 200 for quick sanity check

# Init CUDA
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init("cuda")

# Mixed precision context (required because embeddings may be bf16 while some weights are fp32)
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

try:
    model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=MODEL_TAG, step=MODEL_STEP)
    print0(f"Loaded base checkpoint: tag={MODEL_TAG} step={meta['step']}")

    tasks = {
        "XNLI-TR": XNLI_TR(split="test"),
        "XCOPA-TR": XCOPA_TR(split="test"),
        # Belebele 'tur_Latn' only exposes 'test' in your environment
        "Belebele-TR": Belebele_TR(split="test"),
    }

    for name, task in tasks.items():
        with autocast_ctx:
            acc = run_categorical_eval(
                task_object=task,
                tokenizer=tokenizer,
                model=model,
                batch_size=BATCH_SIZE,
                max_problems=MAX_PROBLEMS,
            )
        print0(f"{name}: {acc*100:.2f}%")

finally:
    compute_cleanup()