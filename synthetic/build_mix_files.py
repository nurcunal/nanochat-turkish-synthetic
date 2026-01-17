"""
Build deterministic 0/25/50/75/100% synthetic training mixes as CustomJSON JSONL files.

Why this exists:
- `scripts/mid_train.py` / `scripts/chat_sft.py` can train from a single CustomJSON file via:
    NANOCHAT_TR_CUSTOMJSON_TRAIN_FILE=/path/to/train.jsonl
    NANOCHAT_TR_CUSTOMJSON_VAL_FILE=/path/to/val.jsonl   (optional)
- For a sweep over synthetic ratios, it's convenient to materialize the mixes as files and then
  loop over ratios, saving checkpoints under distinct `--save-model-tag` folder names.

Output format:
- Each JSONL line is a JSON ARRAY of messages (nanochat CustomJSON):
    [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]

This script does NOT call any external APIs.
It samples from:
- Original (non-synthetic) Turkish tasks via the repo's `tasks/*.py` wrappers (HuggingFace datasets).
- Judge-accepted synthetic TRAIN corpora under `synthetic/accepted_synthetic_datasets/**`.

Typical Colab usage:
  python3 synthetic/build_mix_files.py --stage mid --out-dir synthetic/mixes/mid --n-train 50000 --n-val 2000
  python3 synthetic/build_mix_files.py --stage sft --out-dir synthetic/mixes/sft --n-train 20000 --n-val 2000 --sft-val-size 2000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SYN_DIR = REPO_ROOT / "synthetic"


def _ensure_nanochat_turkish_import_path() -> None:
    """
    Ensure `import tasks.*` works when this script is run from repo root in environments like Colab.

    This repo can appear in either of these layouts:

    - Layout A (this workspace): `tasks/` lives under:
        <repo>/nanochat-master-turkish/tasks
      so we need `<repo>/nanochat-master-turkish` on `sys.path`.

    - Layout B (some Colab clones): `tasks/` lives directly under:
        <repo>/tasks
      so we need `<repo>` on `sys.path`.
    """
    candidates = [
        REPO_ROOT / "nanochat-master-turkish",
        # fallback: sometimes users run from inside nanochat-master-turkish already
        REPO_ROOT,
    ]
    for c in candidates:
        if (c / "tasks").is_dir():
            p = str(c)
            if p not in sys.path:
                sys.path.insert(0, p)
            return


def _read_jsonl_arrays(path: Path) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if not isinstance(obj, list):
                raise ValueError(f"{path}: expected each line to be a JSON array, got {type(obj).__name__}")
            rows.append(obj)
    return rows


def _write_jsonl_arrays(path: Path, rows: list[list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _is_valid_customjson_conversation(msgs: Any) -> tuple[bool, str]:
    if not isinstance(msgs, list) or len(msgs) < 2:
        return False, "not a list or too short"
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            return False, f"msg {i} not a dict"
        role = m.get("role")
        content = m.get("content")
        expected_role = "user" if i % 2 == 0 else "assistant"
        if role != expected_role:
            return False, f"msg {i} role {role!r} != {expected_role!r}"
        if not isinstance(content, str) or not content.strip():
            return False, f"msg {i} content empty"
    return True, "ok"


def _normalize_messages_to_customjson(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert a generic chat message list (may include `system` and/or consecutive same-role messages)
    into nanochat CustomJSON constraints:
    - no `system` messages
    - strict alternation user/assistant starting with user
    - non-empty content for all messages

    Policy:
    - Any `system` messages are folded into the first `user` message as a prefix.
    - Consecutive same-role messages are merged by concatenating with a blank line.
    - If we still can't obtain user/assistant alternation starting with user and ending with assistant,
      we reject by raising ValueError.
    """
    if not isinstance(msgs, list):
        raise ValueError("msgs must be a list")

    system_chunks: list[str] = []
    filtered: list[dict[str, Any]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in {"system", "user", "assistant"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "system":
            system_chunks.append(content.strip())
        else:
            filtered.append({"role": role, "content": content.strip()})

    if not filtered:
        raise ValueError("No non-system messages after filtering")

    # Merge consecutive same-role messages
    merged: list[dict[str, Any]] = []
    for m in filtered:
        if not merged or merged[-1]["role"] != m["role"]:
            merged.append(m)
        else:
            merged[-1]["content"] = (merged[-1]["content"] + "\n\n" + m["content"]).strip()

    sys_prefix = "\n\n".join(system_chunks).strip() if system_chunks else ""

    # If the first non-system role is assistant (assistant-first conversations), salvage them by
    # prepending a synthetic user message. Prefer using the system prompt (if present) as the user content.
    # If no system prompt exists, we fail and let the caller resample (better than injecting junk).
    if merged[0]["role"] == "assistant":
        if not sys_prefix:
            raise ValueError("assistant-first conversation with no system prompt")
        merged = [{"role": "user", "content": sys_prefix}, *merged]

    # Fold system into first user (standard case)
    if sys_prefix and merged[0]["role"] == "user":
        merged[0]["content"] = (sys_prefix + "\n\n" + merged[0]["content"]).strip()

    # Enforce strict alternation and end on assistant
    if merged and merged[-1]["role"] != "assistant":
        # If we end on a user message, drop it (cannot be used for supervised next-token chat loss).
        if merged[-1]["role"] == "user":
            merged = merged[:-1]
    ok, why = _is_valid_customjson_conversation(merged)
    if not ok:
        raise ValueError(why)
    return merged


def _extract_messages(example: Any) -> list[dict[str, Any]]:
    # Tasks return dicts like {"messages":[...], ...}; CustomJSON files store messages arrays directly.
    if isinstance(example, list):
        msgs = example
    elif isinstance(example, dict) and "messages" in example:
        msgs = example["messages"]
    else:
        raise TypeError(f"Unsupported example type: {type(example).__name__}")
    # Normalize any system messages / consecutive role duplicates into strict CustomJSON format.
    return _normalize_messages_to_customjson(msgs)


@dataclass(frozen=True)
class PoolTask:
    name: str
    task: Any  # a Task object
    weight: float


def _load_synthetic_pool(accepted_dir: Path) -> list[list[dict[str, Any]]]:
    files = sorted(accepted_dir.glob("*/*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No accepted synthetic TRAIN jsonl files found under {accepted_dir}")
    pool: list[list[dict[str, Any]]] = []
    for p in files:
        pool.extend(_read_jsonl_arrays(p))
    # validate once
    for msgs in pool:
        ok, why = _is_valid_customjson_conversation(msgs)
        if not ok:
            raise ValueError(f"Synthetic pool has invalid conversation from {accepted_dir}: {why}")
    return pool


def _build_original_tasks_for_stage(stage: str, mmlu_tr_dataset_id: str, xnli_stop: int, mmlu_stop: int, sft_val_size: int):
    # Import lazily so that running this script in environments without the repo on PYTHONPATH
    # fails with a clear error at runtime.
    _ensure_nanochat_turkish_import_path()
    from tasks.turkish_instruct import SungurTR, TurkishSFTV1
    from tasks.xnli_tr import XNLI_TR
    from tasks.mmlu_tr import MMLU_TR

    tasks: list[PoolTask] = []
    if stage == "mid":
        t1 = SungurTR(split="train")
        t2 = TurkishSFTV1(split="train")
        t3 = XNLI_TR(split="train", stop=xnli_stop)
        t4 = MMLU_TR(split="train", dataset_id=mmlu_tr_dataset_id, stop=mmlu_stop)
        # weights proportional to dataset lengths (approx baseline composition)
        tasks = [
            PoolTask("SungurTR(train)", t1, float(len(t1))),
            PoolTask("TurkishSFTV1(train)", t2, float(len(t2))),
            PoolTask(f"XNLI_TR(train,stop={xnli_stop})", t3, float(len(t3))),
            PoolTask(f"MMLU_TR(train,stop={mmlu_stop})", t4, float(len(t4))),
        ]
        # Validation pool (original-only): small slices similar to mid_train defaults
        v1 = XNLI_TR(split="validation", stop=min(5000, xnli_stop))
        v2 = MMLU_TR(split="validation", dataset_id=mmlu_tr_dataset_id, stop=min(2000, mmlu_stop))
        v3 = TurkishSFTV1(split="train", stop=2000)
        val_tasks = [
            PoolTask("XNLI_TR(validation)", v1, float(len(v1))),
            PoolTask("MMLU_TR(validation)", v2, float(len(v2))),
            PoolTask("TurkishSFTV1(train,stop=2000)", v3, float(len(v3))),
        ]
        return tasks, val_tasks

    if stage == "sft":
        # Mimic chat_sft.py: reserve first `sft_val_size` examples for val.
        base_train = TurkishSFTV1(split="train", start=sft_val_size)
        base_val = TurkishSFTV1(split="train", stop=sft_val_size)
        t_mmlu = MMLU_TR(split="train", dataset_id=mmlu_tr_dataset_id, stop=mmlu_stop)
        tasks = [
            PoolTask(f"TurkishSFTV1(train,start={sft_val_size})", base_train, float(len(base_train))),
            PoolTask(f"MMLU_TR(train,stop={mmlu_stop})", t_mmlu, float(len(t_mmlu))),
        ]
        val_tasks = [
            PoolTask(f"TurkishSFTV1(train,stop={sft_val_size})", base_val, float(len(base_val))),
        ]
        return tasks, val_tasks

    raise ValueError(f"Unknown stage: {stage}")


def _weighted_choice(rng: random.Random, items: list[PoolTask]) -> PoolTask:
    total = sum(max(0.0, it.weight) for it in items)
    if total <= 0:
        return items[rng.randrange(len(items))]
    r = rng.random() * total
    acc = 0.0
    for it in items:
        acc += max(0.0, it.weight)
        if r <= acc:
            return it
    return items[-1]


def _sample_from_tasks(rng: random.Random, tasks: list[PoolTask], n: int) -> list[list[dict[str, Any]]]:
    out: list[list[dict[str, Any]]] = []
    for _ in range(n):
        # Some rows may not be normalizable into strict CustomJSON (e.g., assistant-first without system).
        # Retry a few times before failing.
        last_err: Exception | None = None
        for _try in range(50):
            t = _weighted_choice(rng, tasks)
            idx = rng.randrange(len(t.task))
            ex = t.task[idx]
            try:
                out.append(_extract_messages(ex))
                last_err = None
                break
            except Exception as e:
                last_err = e
                continue
        if last_err is not None:
            raise RuntimeError(f"Failed to sample a normalizable conversation after retries: {last_err}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["mid", "sft"])
    ap.add_argument("--out-dir", required=True, help="Output directory (will be created).")
    ap.add_argument("--ratios", type=str, default="0,25,50,75,100", help="Comma-separated synthetic % ratios.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=50_000, help="Total training examples per mix (examples, not tokens).")
    ap.add_argument("--n-val", type=int, default=2_000, help="Validation examples (original-only val file).")
    ap.add_argument("--sft-val-size", type=int, default=2000, help="For stage=sft: matches chat_sft TurkishSFTV1 split.")
    ap.add_argument("--xnli-stop", type=int, default=50_000)
    ap.add_argument("--mmlu-stop", type=int, default=50_000)
    ap.add_argument("--mmlu-tr-dataset-id", type=str, default="malhajar/mmlu_tr-v0.2")
    ap.add_argument("--accepted-synth-train-dir", type=str, default=str(SYN_DIR / "accepted_synthetic_datasets"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ratios = [int(x.strip()) for x in args.ratios.split(",") if x.strip() != ""]
    for r in ratios:
        if r < 0 or r > 100:
            raise ValueError(f"ratio out of range: {r}")

    # Pools
    synth_pool = _load_synthetic_pool(Path(args.accepted_synth_train_dir))
    orig_tasks, val_tasks = _build_original_tasks_for_stage(
        stage=args.stage,
        mmlu_tr_dataset_id=args.mmlu_tr_dataset_id,
        xnli_stop=args.xnli_stop,
        mmlu_stop=args.mmlu_stop,
        sft_val_size=args.sft_val_size,
    )

    # Build a single original-only validation file (shared across ratios)
    rng_val = random.Random(args.seed + 12345)
    val_rows = _sample_from_tasks(rng_val, val_tasks, args.n_val)
    val_path = out_dir / "val_original.jsonl"
    _write_jsonl_arrays(val_path, val_rows)

    # Also write a synthetic-only validation file (optional diagnostic)
    rng_val_syn = random.Random(args.seed + 54321)
    syn_val_rows = [synth_pool[rng_val_syn.randrange(len(synth_pool))] for _ in range(min(args.n_val, len(synth_pool)))]
    syn_val_path = out_dir / "val_synth_only.jsonl"
    _write_jsonl_arrays(syn_val_path, syn_val_rows)

    # Mix files
    manifest: dict[str, Any] = {
        "stage": args.stage,
        "seed": args.seed,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "ratios": ratios,
        "val_original": str(val_path),
        "val_synth_only": str(syn_val_path),
        "accepted_synth_train_dir": args.accepted_synth_train_dir,
        "original_tasks": [t.name for t in orig_tasks],
    }

    for r in ratios:
        n_syn = int(round(args.n_train * (r / 100.0)))
        n_orig = args.n_train - n_syn
        rng = random.Random(args.seed + r)

        orig_rows = _sample_from_tasks(rng, orig_tasks, n_orig)
        syn_rows = [synth_pool[rng.randrange(len(synth_pool))] for _ in range(n_syn)]

        all_rows = orig_rows + syn_rows
        rng.shuffle(all_rows)

        out_path = out_dir / f"mix_{r}syn_train.jsonl"
        _write_jsonl_arrays(out_path, all_rows)
        manifest[f"mix_{r}syn_train"] = {"path": str(out_path), "n_orig": n_orig, "n_syn": n_syn, "n_total": len(all_rows)}

        print(f"[OK] {args.stage}: {out_path} (orig={n_orig}, syn={n_syn})")

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

