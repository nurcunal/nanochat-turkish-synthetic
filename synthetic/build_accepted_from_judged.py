"""
Build accepted synthetic corpora from existing judge artifacts (NO API calls).

This script assumes you already have judge outputs under:
  synthetic/llm-judge/**/_judged.jsonl

It will:
1) Reconstruct FULL accepted/rejected partitions from the original synthetic JSONL inputs
   using the judge `results` list (per-item index + accept flag).
   This avoids any truncation issues in `accepted_items.items`.
2) Patch each judged artifact to include:
   - judge_model_label (e.g., "gemini-3-pro")
   - source_name (dataset/benchmark name)
   - synthetic_path (input file path used for reconstruction)
3) Write merged training-ready JSONL outputs:
   - synthetic/accepted_synthetic_datasets/<group>/*.jsonl
   - synthetic/accepted_synthetic_benchmark_evals/<group>/*.jsonl

Format guarantee:
- Output JSONL lines are JSON arrays of messages (CustomJSON), compatible with:
  nanochat-master-turkish/tasks/customjson.py

IMPORTANT:
- The accepted benchmark eval sets are for EVALUATION ONLY and must not be used for training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SYN = REPO_ROOT / "synthetic"


def _read_jsonl(path: Path) -> list[Any]:
    out: list[Any] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(json.loads(s))
    return out


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _is_valid_customjson_conversation(x: Any) -> tuple[bool, str]:
    if not isinstance(x, list):
        return False, f"Expected list, got {type(x).__name__}"
    if len(x) < 2:
        return False, "Conversation must have >=2 messages"
    for i, m in enumerate(x):
        if not isinstance(m, dict):
            return False, f"Message {i} not a dict"
        role = m.get("role")
        content = m.get("content")
        expected_role = "user" if i % 2 == 0 else "assistant"
        if role != expected_role:
            return False, f"Message {i} role={role!r} expected={expected_role!r}"
        if not isinstance(content, str) or content.strip() == "":
            return False, f"Message {i} content empty or not a string"
    return True, "ok"


@dataclass(frozen=True)
class Mapping:
    judged_path: Path
    synthetic_path: Path
    source_name: str
    group: str  # subfolder name
    kind: str   # "dataset" | "benchmark"
    out_accepted_name: str


MAPPINGS: list[Mapping] = [
    # TRAIN datasets
    Mapping(
        judged_path=SYN / "llm-judge" / "synthetic_datasets" / "synthetic_train_sungurtr_100_judged.jsonl",
        synthetic_path=SYN / "synthetic_datasets" / "sungurtr" / "synthetic_train_sungurtr_100.jsonl",
        source_name="SungurTR-style synthetic TRAIN (instruction)",
        group="sungurtr",
        kind="dataset",
        out_accepted_name="synthetic_train_sungurtr_accepted.jsonl",
    ),
    Mapping(
        judged_path=SYN / "llm-judge" / "synthetic_datasets" / "synthetic_train_turkishsftv1_100_judged.jsonl",
        synthetic_path=SYN / "synthetic_datasets" / "turkishsftv1" / "synthetic_train_turkishsftv1_100.jsonl",
        source_name="TurkishSFTV1-style synthetic TRAIN (instruction)",
        group="turkishsftv1",
        kind="dataset",
        out_accepted_name="synthetic_train_turkishsftv1_accepted.jsonl",
    ),
    Mapping(
        judged_path=SYN / "llm-judge" / "synthetic_datasets" / "synthetic_train_mmlu_tr_100_judged.jsonl",
        synthetic_path=SYN / "synthetic_datasets" / "mmlu_tr" / "synthetic_train_mmlu_tr_100.jsonl",
        source_name="MMLU-TR-like synthetic TRAIN (MCQ)",
        group="mmlu_tr",
        kind="dataset",
        out_accepted_name="synthetic_train_mmlu_tr_accepted.jsonl",
    ),
    # EVAL benchmarks
    Mapping(
        judged_path=SYN / "llm-judge" / "synthetic_benchmarks" / "synthetic_eval_xnli_100_judged.jsonl",
        synthetic_path=SYN / "synthetic_benchmarks" / "xnli_tr" / "synthetic_eval_xnli_100.jsonl",
        source_name="XNLI-like synthetic EVAL (3-way NLI, held-out)",
        group="xnli_tr",
        kind="benchmark",
        out_accepted_name="synthetic_eval_xnli_accepted.jsonl",
    ),
    Mapping(
        judged_path=SYN / "llm-judge" / "synthetic_benchmarks" / "synthetic_eval_xcopa_like_100_judged.jsonl",
        synthetic_path=SYN / "synthetic_benchmarks" / "xcopa_tr" / "synthetic_eval_xcopa_like_100.jsonl",
        source_name="XCOPA-like synthetic EVAL (2-way causal, held-out)",
        group="xcopa_tr",
        kind="benchmark",
        out_accepted_name="synthetic_eval_xcopa_like_accepted.jsonl",
    ),
    Mapping(
        judged_path=SYN / "llm-judge" / "synthetic_benchmarks" / "synthetic_eval_belebele_like_100_judged.jsonl",
        synthetic_path=SYN / "synthetic_benchmarks" / "belebele_tr" / "synthetic_eval_belebele_like_100.jsonl",
        source_name="Belebele-like synthetic EVAL (4-way RC, held-out)",
        group="belebele_tr",
        kind="benchmark",
        out_accepted_name="synthetic_eval_belebele_like_accepted.jsonl",
    ),
    Mapping(
        judged_path=SYN / "llm-judge" / "synthetic_benchmarks" / "synthetic_eval_plu_like_100_judged.jsonl",
        synthetic_path=SYN / "synthetic_benchmarks" / "plu_tr" / "synthetic_eval_plu_100.jsonl",
        source_name="Turkish-PLU-like synthetic EVAL (4-way next-event, held-out)",
        group="plu_tr",
        kind="benchmark",
        out_accepted_name="synthetic_eval_plu_accepted.jsonl",
    ),
    Mapping(
        judged_path=SYN / "llm-judge" / "synthetic_benchmarks" / "synthetic_eval_turkishmmlu_100_judged.jsonl",
        synthetic_path=SYN / "synthetic_benchmarks" / "mmlu_tr" / "synthetic_eval_turkishmmlu_100.jsonl",
        source_name="TurkishMMLU-like synthetic EVAL (4-way MCQ, held-out)",
        group="mmlu_tr",
        kind="benchmark",
        out_accepted_name="synthetic_eval_turkishmmlu_accepted.jsonl",
    ),
]


def _load_judged_json(path: Path) -> dict[str, Any]:
    txt = path.read_text(encoding="utf-8")
    if not txt.strip():
        raise RuntimeError(
            f"{path} is empty on disk.\n"
            "If you see content in the editor, please SAVE the file so this script can read it."
        )
    try:
        obj = json.loads(txt)
        if not isinstance(obj, dict) or "results" not in obj:
            raise ValueError(f"{path} must be a single JSON object with key 'results'")
        return obj
    except json.JSONDecodeError:
        # Some judged artifacts may contain literal truncation placeholders (e.g. "... (truncated for brevity)")
        # inside accepted_items/rejected_items, which makes them invalid JSON. We don't need those blocks anyway:
        # we reconstruct full partitions from `results` + the original synthetic JSONL.
        start = txt.find('"accepted_items"')
        if start == -1:
            raise

        # Keep everything before accepted_items (should include summary + results and the trailing comma after results),
        # then replace the rest with minimal valid stubs. We'll overwrite stubs later.
        prefix = txt[:start].rstrip()
        if not prefix.endswith(","):
            # If the prefix doesn't end with a comma, ensure we separate fields correctly.
            prefix = prefix + ","
        sanitized = (
            prefix
            + '\n  "accepted_items": {"items": []},\n  "rejected_items": {"items": []}\n}\n'
        )
        obj2 = json.loads(sanitized)
        if not isinstance(obj2, dict) or "results" not in obj2:
            raise ValueError(f"{path}: failed to recover a judged JSON object with key 'results'")
        return obj2


def _reconstruct_partitions(judged: dict[str, Any], synthetic_rows: list[Any]) -> tuple[list[Any], list[Any]]:
    results = judged.get("results", [])
    if not isinstance(results, list):
        raise ValueError("judged['results'] must be a list")
    accept_by_index: dict[int, bool] = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        idx = r.get("index")
        if not isinstance(idx, int):
            continue
        accept_by_index[idx] = bool(r.get("accept", False))

    accepted: list[Any] = []
    rejected: list[Any] = []
    for i, conv in enumerate(synthetic_rows):
        ok, why = _is_valid_customjson_conversation(conv)
        if not ok:
            # invalid schema => always rejected
            rejected.append(conv)
            continue
        if accept_by_index.get(i, False):
            accepted.append(conv)
        else:
            rejected.append(conv)
    return accepted, rejected


def main() -> None:
    judge_model_label = (os.environ.get("JUDGE_MODEL_LABEL", "").strip() or "gemini-3-pro")

    out_train_root = SYN / "accepted_synthetic_datasets"
    out_eval_root = SYN / "accepted_synthetic_benchmark_evals"

    for m in MAPPINGS:
        judged = _load_judged_json(m.judged_path)
        synthetic_rows = _read_jsonl(m.synthetic_path)
        accepted, rejected = _reconstruct_partitions(judged, synthetic_rows)

        # Patch judged artifact to ensure non-truncated full partitions + metadata
        summary = judged.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        summary.update(
            {
                "judge_model_label": judge_model_label,
                "source_name": m.source_name,
                "synthetic_path": str(m.synthetic_path),
                "n_total": len(synthetic_rows),
                "n_accepted": len(accepted),
                "accept_rate": (len(accepted) / max(1, len(synthetic_rows))),
            }
        )
        judged["summary"] = summary
        judged["accepted_items"] = {"items": accepted}
        judged["rejected_items"] = {"items": rejected}
        _write_json(m.judged_path, judged)

        # Write accepted JSONL to the right folder
        out_dir = out_train_root / m.group if m.kind == "dataset" else out_eval_root / m.group
        out_path = out_dir / m.out_accepted_name
        _write_jsonl(out_path, accepted)

        print(f"[OK] {m.judged_path.name}: accepted {len(accepted)}/{len(synthetic_rows)} -> {out_path}")


if __name__ == "__main__":
    import os
    main()

