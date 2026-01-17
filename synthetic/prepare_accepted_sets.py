"""
Prepare "accepted synthetic" corpora for training and evaluation.

Why this exists:
- The project rubric requires synthetic data generation + quality validation.
- We validate synthetic artifacts with `scripts/synthetic_judge_gemini.py`, which can emit:
  - accepted.jsonl / rejected.jsonl (CustomJSON JSONL: each line is a JSON array of messages)
  - a combined `*_judged.jsonl` JSON artifact with full accepted/rejected partitions.

This script runs the judge over all synthetic files in `synthetic/` and writes:
- synthetic/llm-judge/.../*_judged.jsonl   (single JSON object; auditable)
- synthetic/accepted_synthetic_datasets/.../*.jsonl
- synthetic/accepted_synthetic_benchmark_evals/.../*.jsonl

It also validates that accepted outputs are compatible with `tasks/customjson.py`:
- each JSONL line must be a JSON array of messages
- roles alternate user/assistant starting with user
- content fields are non-empty strings

Usage (from repo root):
  export GEMINI_API_KEY=...
  export GEMINI_MODEL=...              # API model id you have access to
  export JUDGE_MODEL_LABEL=gemini-3-pro
  python3 synthetic/prepare_accepted_sets.py
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = REPO_ROOT / "synthetic"


def _read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


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


def _validate_customjson_file(path: Path) -> None:
    rows = _read_jsonl(path)
    if not rows:
        raise RuntimeError(f"{path} is empty")
    for i, r in enumerate(rows[:50]):  # sample a prefix for fast feedback
        ok, why = _is_valid_customjson_conversation(r)
        if not ok:
            raise RuntimeError(f"{path}: invalid row {i}: {why}")


@dataclass(frozen=True)
class Item:
    # input synthetic file
    synthetic_path: Path
    # output judged artifact
    out_judged: Path
    # output accepted file (CustomJSON JSONL)
    out_accepted: Path
    # output rejected file (CustomJSON JSONL)
    out_rejected: Path
    # output report JSON
    out_report: Path
    # judge metadata
    task_hint: str
    source_name: str


def _judge_one(it: Item) -> None:
    it.out_judged.parent.mkdir(parents=True, exist_ok=True)
    it.out_accepted.parent.mkdir(parents=True, exist_ok=True)
    it.out_rejected.parent.mkdir(parents=True, exist_ok=True)
    it.out_report.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3",
        str(REPO_ROOT / "nanochat-master-turkish" / "scripts" / "synthetic_judge_gemini.py"),
        "--synthetic",
        str(it.synthetic_path),
        "--task-hint",
        it.task_hint,
        "--source-name",
        it.source_name,
        "--out-accepted",
        str(it.out_accepted),
        "--out-rejected",
        str(it.out_rejected),
        "--out-report",
        str(it.out_report),
        "--out-judged",
        str(it.out_judged),
    ]
    subprocess.run(cmd, check=True)
    _validate_customjson_file(it.out_accepted)


def main() -> None:
    # Synthetic TRAIN datasets
    train_items = [
        Item(
            synthetic_path=SYNTHETIC_DIR / "synthetic_datasets" / "sungurtr" / "synthetic_train_sungurtr_100.jsonl",
            out_judged=SYNTHETIC_DIR / "llm-judge" / "synthetic_datasets" / "synthetic_train_sungurtr_100_judged.jsonl",
            out_accepted=SYNTHETIC_DIR / "accepted_synthetic_datasets" / "sungurtr" / "synthetic_train_sungurtr_accepted.jsonl",
            out_rejected=SYNTHETIC_DIR / "llm-judge" / "rejected" / "synthetic_datasets" / "synthetic_train_sungurtr_rejected.jsonl",
            out_report=SYNTHETIC_DIR / "llm-judge" / "reports" / "synthetic_datasets" / "synthetic_train_sungurtr_report.json",
            task_hint="train_sungurtr",
            source_name="SungurTR-style synthetic TRAIN (instruction)",
        ),
        Item(
            synthetic_path=SYNTHETIC_DIR / "synthetic_datasets" / "turkishsftv1" / "synthetic_train_turkishsftv1_100.jsonl",
            out_judged=SYNTHETIC_DIR / "llm-judge" / "synthetic_datasets" / "synthetic_train_turkishsftv1_100_judged.jsonl",
            out_accepted=SYNTHETIC_DIR / "accepted_synthetic_datasets" / "turkishsftv1" / "synthetic_train_turkishsftv1_accepted.jsonl",
            out_rejected=SYNTHETIC_DIR / "llm-judge" / "rejected" / "synthetic_datasets" / "synthetic_train_turkishsftv1_rejected.jsonl",
            out_report=SYNTHETIC_DIR / "llm-judge" / "reports" / "synthetic_datasets" / "synthetic_train_turkishsftv1_report.json",
            task_hint="train_turkishsftv1",
            source_name="TurkishSFTV1-style synthetic TRAIN (instruction)",
        ),
        Item(
            synthetic_path=SYNTHETIC_DIR / "synthetic_datasets" / "mmlu_tr" / "synthetic_train_mmlu_tr_100.jsonl",
            out_judged=SYNTHETIC_DIR / "llm-judge" / "synthetic_datasets" / "synthetic_train_mmlu_tr_100_judged.jsonl",
            out_accepted=SYNTHETIC_DIR / "accepted_synthetic_datasets" / "mmlu_tr" / "synthetic_train_mmlu_tr_accepted.jsonl",
            out_rejected=SYNTHETIC_DIR / "llm-judge" / "rejected" / "synthetic_datasets" / "synthetic_train_mmlu_tr_rejected.jsonl",
            out_report=SYNTHETIC_DIR / "llm-judge" / "reports" / "synthetic_datasets" / "synthetic_train_mmlu_tr_report.json",
            task_hint="train_mmlu_tr",
            source_name="MMLU-TR-like synthetic TRAIN (MCQ)",
        ),
    ]

    # Synthetic EVAL benchmarks (held-out)
    eval_items = [
        Item(
            synthetic_path=SYNTHETIC_DIR / "synthetic_benchmarks" / "xnli_tr" / "synthetic_eval_xnli_100.jsonl",
            out_judged=SYNTHETIC_DIR / "llm-judge" / "synthetic_benchmarks" / "synthetic_eval_xnli_100_judged.jsonl",
            out_accepted=SYNTHETIC_DIR / "accepted_synthetic_benchmark_evals" / "xnli_tr" / "synthetic_eval_xnli_accepted.jsonl",
            out_rejected=SYNTHETIC_DIR / "llm-judge" / "rejected" / "synthetic_benchmarks" / "synthetic_eval_xnli_rejected.jsonl",
            out_report=SYNTHETIC_DIR / "llm-judge" / "reports" / "synthetic_benchmarks" / "synthetic_eval_xnli_report.json",
            task_hint="bench_xnli_like",
            source_name="XNLI-like synthetic EVAL (3-way NLI, held-out)",
        ),
        Item(
            synthetic_path=SYNTHETIC_DIR / "synthetic_benchmarks" / "xcopa_tr" / "synthetic_eval_xcopa_like_100.jsonl",
            out_judged=SYNTHETIC_DIR / "llm-judge" / "synthetic_benchmarks" / "synthetic_eval_xcopa_like_100_judged.jsonl",
            out_accepted=SYNTHETIC_DIR / "accepted_synthetic_benchmark_evals" / "xcopa_tr" / "synthetic_eval_xcopa_like_accepted.jsonl",
            out_rejected=SYNTHETIC_DIR / "llm-judge" / "rejected" / "synthetic_benchmarks" / "synthetic_eval_xcopa_like_rejected.jsonl",
            out_report=SYNTHETIC_DIR / "llm-judge" / "reports" / "synthetic_benchmarks" / "synthetic_eval_xcopa_like_report.json",
            task_hint="bench_xcopa_like",
            source_name="XCOPA-like synthetic EVAL (2-way causal, held-out)",
        ),
        Item(
            synthetic_path=SYNTHETIC_DIR / "synthetic_benchmarks" / "belebele_tr" / "synthetic_eval_belebele_like_100.jsonl",
            out_judged=SYNTHETIC_DIR / "llm-judge" / "synthetic_benchmarks" / "synthetic_eval_belebele_like_100_judged.jsonl",
            out_accepted=SYNTHETIC_DIR / "accepted_synthetic_benchmark_evals" / "belebele_tr" / "synthetic_eval_belebele_like_accepted.jsonl",
            out_rejected=SYNTHETIC_DIR / "llm-judge" / "rejected" / "synthetic_benchmarks" / "synthetic_eval_belebele_like_rejected.jsonl",
            out_report=SYNTHETIC_DIR / "llm-judge" / "reports" / "synthetic_benchmarks" / "synthetic_eval_belebele_like_report.json",
            task_hint="bench_belebele_like",
            source_name="Belebele-like synthetic EVAL (4-way RC, held-out)",
        ),
        Item(
            synthetic_path=SYNTHETIC_DIR / "synthetic_benchmarks" / "plu_tr" / "synthetic_eval_plu_100.jsonl",
            out_judged=SYNTHETIC_DIR / "llm-judge" / "synthetic_benchmarks" / "synthetic_eval_plu_like_100_judged.jsonl",
            out_accepted=SYNTHETIC_DIR / "accepted_synthetic_benchmark_evals" / "plu_tr" / "synthetic_eval_plu_accepted.jsonl",
            out_rejected=SYNTHETIC_DIR / "llm-judge" / "rejected" / "synthetic_benchmarks" / "synthetic_eval_plu_rejected.jsonl",
            out_report=SYNTHETIC_DIR / "llm-judge" / "reports" / "synthetic_benchmarks" / "synthetic_eval_plu_report.json",
            task_hint="bench_plu_like",
            source_name="Turkish-PLU-like synthetic EVAL (4-way next-event, held-out)",
        ),
        Item(
            synthetic_path=SYNTHETIC_DIR / "synthetic_benchmarks" / "mmlu_tr" / "synthetic_eval_turkishmmlu_100.jsonl",
            out_judged=SYNTHETIC_DIR / "llm-judge" / "synthetic_benchmarks" / "synthetic_eval_turkishmmlu_100_judged.jsonl",
            out_accepted=SYNTHETIC_DIR / "accepted_synthetic_benchmark_evals" / "mmlu_tr" / "synthetic_eval_turkishmmlu_accepted.jsonl",
            out_rejected=SYNTHETIC_DIR / "llm-judge" / "rejected" / "synthetic_benchmarks" / "synthetic_eval_turkishmmlu_rejected.jsonl",
            out_report=SYNTHETIC_DIR / "llm-judge" / "reports" / "synthetic_benchmarks" / "synthetic_eval_turkishmmlu_report.json",
            task_hint="bench_turkishmmlu",
            source_name="TurkishMMLU-like synthetic EVAL (4-way MCQ, held-out)",
        ),
    ]

    missing = [it.synthetic_path for it in [*train_items, *eval_items] if not it.synthetic_path.exists()]
    if missing:
        raise SystemExit("Missing synthetic inputs:\n" + "\n".join(str(p) for p in missing))

    print("== Judging synthetic TRAIN datasets ==")
    for it in train_items:
        print(f"- {it.synthetic_path.name} -> {it.out_accepted}")
        _judge_one(it)

    print("== Judging synthetic EVAL benchmarks ==")
    for it in eval_items:
        print(f"- {it.synthetic_path.name} -> {it.out_accepted}")
        _judge_one(it)

    print("Done. Accepted corpora written under:")
    print(f"- {SYNTHETIC_DIR / 'accepted_synthetic_datasets'}")
    print(f"- {SYNTHETIC_DIR / 'accepted_synthetic_benchmark_evals'}")


if __name__ == "__main__":
    main()

