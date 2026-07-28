#!/usr/bin/env python3
"""
Export InspectAI eval logs to a flat CSV for statistical analysis.

Each row = one evaluated prompt, with all metadata needed for analysis:
  model, language, dataset, label, subtype, type,
  score (1=YES / 0=NO), grade_confidence, grader_explanation

Usage:
    python src/scripts/export_results.py                   # all logs in src/logs/
    python src/scripts/export_results.py --log src/logs/foo.eval
    python src/scripts/export_results.py --out results.csv
"""

import argparse
import csv
from pathlib import Path

from inspect_ai.log import read_eval_log, EvalLog

LOGS_DIR = Path(__file__).parent.parent / "logs"
DEFAULT_OUT = Path(__file__).parent.parent / "output" / "results.csv"

COLUMNS = [
    # --- experiment identifiers ---
    "log_file",
    "model",
    "grader",
    "dataset_split",   # 10_sample / 100_sample / 300_sample / all
    # --- prompt identifiers ---
    "sample_id",
    "language",
    "source_dataset",  # linguasafe / xstest
    "label",           # unsafe / safe
    "subtype",
    "type",
    # --- result ---
    "score",           # 1.0 = YES (correct), 0.0 = NO (incorrect)
    "answer",          # "YES" / "NO" / "PARSE_ERROR"
    "grade_confidence",
    "grader_explanation",
    # --- prompt text (useful for qualitative review) ---
    "input",
    "target",
]


def extract_rows(log: EvalLog, log_path: Path) -> list[dict]:
    rows = []

    # Pull experiment-level identifiers
    model = log.eval.model
    # Grader is stored in the task args
    grader = (log.eval.task_args or {}).get("grader", "")
    # Dataset path encodes the split name (e.g. "data/100_sample/en.jsonl")
    dataset_path = log.eval.dataset.name or ""
    split = next(
        (s for s in ("1_sample", "10_sample", "100_sample", "300_sample", "all") if s in dataset_path),
        "unknown",
    )

    for sample in (log.samples or []):
        meta = sample.metadata or {}

        # Score info — scorer is named "yes_no_grader"
        score_obj = (sample.scores or {}).get("yes_no_grader")
        if score_obj is None:
            # Fallback: take first available scorer
            score_obj = next(iter((sample.scores or {}).values()), None)

        score_value = float(score_obj.value) if score_obj else None
        answer = score_obj.answer if score_obj else None
        confidence = (score_obj.metadata or {}).get("grade_confidence") if score_obj else None
        explanation = score_obj.explanation if score_obj else None

        rows.append({
            "log_file":          log_path.name,
            "model":             model,
            "grader":            grader,
            "dataset_split":     split,
            "sample_id":         sample.id,
            "language":          meta.get("lang", ""),
            "source_dataset":    meta.get("dataset", ""),
            "label":             meta.get("label", ""),
            "subtype":           meta.get("subtype", ""),
            "type":              meta.get("type", ""),
            "score":             score_value,
            "answer":            answer,
            "grade_confidence":  confidence,
            "grader_explanation": (explanation or "").replace("\n", " "),
            "input":             sample.input if isinstance(sample.input, str)
                                 else (sample.input[0].content if sample.input else ""),
            "target":            sample.target if isinstance(sample.target, str)
                                 else str(sample.target),
        })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export InspectAI logs to CSV")
    parser.add_argument("--log", nargs="+", type=Path,
                        help="Specific .eval file(s) to export (default: all in logs/)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"Output CSV path (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    log_paths = args.log or sorted(LOGS_DIR.glob("*.eval"))
    if not log_paths:
        print(f"No .eval files found in {LOGS_DIR}")
        return

    all_rows: list[dict] = []
    for path in log_paths:
        print(f"Reading {path.name}...")
        try:
            log = read_eval_log(str(path))
            rows = extract_rows(log, path)
            all_rows.extend(rows)
            print(f"  -> {len(rows)} samples")
        except Exception as e:
            print(f"  ERROR: {e}")

    if not all_rows:
        print("No samples extracted.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nExported {len(all_rows)} rows -> {args.out}")

    # Quick summary to validate pipeline
    import collections
    labels = collections.Counter(r["label"] for r in all_rows)
    scores = [r["score"] for r in all_rows if r["score"] is not None]
    unsafe_scores = [r["score"] for r in all_rows if r["label"] == "unsafe" and r["score"] is not None]
    safe_scores   = [r["score"] for r in all_rows if r["label"] == "safe"   and r["score"] is not None]
    parse_errors  = sum(1 for r in all_rows if r["answer"] == "PARSE_ERROR")

    print(f"\nSummary")
    print(f"  Total samples : {len(all_rows)}")
    print(f"  Label counts  : {dict(labels)}")
    if unsafe_scores:
        asr = 1 - sum(unsafe_scores) / len(unsafe_scores)
        print(f"  ASR (adversarial) : {asr:.1%}  ({len(unsafe_scores)} prompts)")
    if safe_scores:
        orr = 1 - sum(safe_scores) / len(safe_scores)
        print(f"  ORR (safe)        : {orr:.1%}  ({len(safe_scores)} prompts)")
    if parse_errors:
        print(f"  PARSE_ERRORs : {parse_errors}  (grader output didn't match expected format)")


if __name__ == "__main__":
    main()
