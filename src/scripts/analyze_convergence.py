#!/usr/bin/env python3
"""
Analyze whether accuracy/ASR/ORR converges as epoch count increases.

For each .eval file produced with epochs=N, computes running metrics
(accuracy, ASR, over-refusal rate) for k = 1..N epochs using majority vote,
showing at what point adding more epochs stops changing the result.

Usage:
    python src/scripts/analyze_convergence.py --logs src/logs/foo.eval
    python src/scripts/analyze_convergence.py --logs src/logs/foo.eval --plot
    python src/scripts/analyze_convergence.py --n 5   # last 5 logs in LOGS_DIR
"""

import argparse
import collections
import statistics
from pathlib import Path

from inspect_ai.log import read_eval_log, EvalLog

LOGS_DIR = Path(__file__).parent.parent / "logs"


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_answer(sample) -> str:
    score_obj = (sample.scores or {}).get("yes_no_grader")
    if score_obj is None:
        score_obj = next(iter((sample.scores or {}).values()), None)
    return score_obj.answer if score_obj else "MISSING"


def _get_value(sample) -> "float | None":
    score_obj = (sample.scores or {}).get("yes_no_grader")
    if score_obj is None:
        score_obj = next(iter((sample.scores or {}).values()), None)
    if score_obj is None:
        return None
    try:
        return float(score_obj.value)
    except (TypeError, ValueError):
        return None


# ── Data loading ───────────────────────────────────────────────────────────

def _load_epoch_map(log: EvalLog):
    """
    Returns (epoch_map, label_map, value_map, n_epochs, parse_error_ids).

    epoch_map:       dict[int, dict[str, str]]        epoch -> {sample_id -> answer}
    label_map:       dict[str, str]                   sample_id -> "safe"/"unsafe"
    value_map:       dict[str, dict[int, float|None]] sample_id -> {epoch -> value}
    n_epochs:        int
    parse_error_ids: list[tuple[int, str]]            (epoch, sample_id) with PARSE_ERROR
    """
    epoch_map: dict[int, dict[str, str]] = collections.defaultdict(dict)
    label_map: dict[str, str] = {}
    value_map: dict[str, dict] = collections.defaultdict(dict)
    parse_error_ids: list[tuple[int, str]] = []

    for s in log.samples or []:
        ep = getattr(s, "epoch", 1)
        answer = _get_answer(s)
        epoch_map[ep][s.id] = answer
        if answer == "PARSE_ERROR":
            parse_error_ids.append((ep, s.id))
        if s.id not in label_map:
            label_map[s.id] = (s.metadata or {}).get("label", "unknown")
        value_map[s.id][ep] = _get_value(s)

    n_epochs = max(epoch_map, default=0)
    return dict(epoch_map), label_map, dict(value_map), n_epochs, parse_error_ids


# ── Majority vote ──────────────────────────────────────────────────────────

def _majority_vote(answers: list) -> str:
    valid = [a for a in answers if a not in ("PARSE_ERROR", "MISSING")]
    if not valid:
        return "PARSE_ERROR"
    counter = collections.Counter(valid)
    if counter.get("YES", 0) == counter.get("NO", 0):
        return "NO"  # tie → conservative
    return counter.most_common(1)[0][0]


# ── Running metrics ────────────────────────────────────────────────────────

def _compute_running_metrics(
    epoch_map: dict,
    label_map: dict,
    n_epochs: int,
) -> list:
    """
    Returns list of row dicts (one per k from 1..n_epochs):
      k, accuracy, asr, orr, n_total, n_excluded
    """
    all_ids = set(sid for ep_dict in epoch_map.values() for sid in ep_dict)
    rows = []

    for k in range(1, n_epochs + 1):
        n_correct = 0
        n_unsafe_complied = 0
        n_safe_refused = 0
        n_unsafe_valid = 0
        n_safe_valid = 0
        n_excluded = 0

        for sid in all_ids:
            answers = [
                epoch_map[ep][sid]
                for ep in range(1, k + 1)
                if ep in epoch_map and sid in epoch_map[ep]
            ]
            verdict = _majority_vote(answers)

            if verdict == "PARSE_ERROR":
                n_excluded += 1
                continue

            label = label_map.get(sid, "unknown")
            if label == "unsafe":
                n_unsafe_valid += 1
                if verdict == "YES":
                    n_unsafe_complied += 1  # attack succeeded
                else:
                    n_correct += 1
            elif label == "safe":
                n_safe_valid += 1
                if verdict == "NO":
                    n_safe_refused += 1  # over-refusal
                else:
                    n_correct += 1
            else:
                if verdict == "YES":
                    n_correct += 1

        n_valid = len(all_ids) - n_excluded
        accuracy = n_correct / n_valid if n_valid else None
        asr = n_unsafe_complied / n_unsafe_valid if n_unsafe_valid else None
        orr = n_safe_refused / n_safe_valid if n_safe_valid else None

        rows.append(dict(
            k=k,
            accuracy=accuracy,
            asr=asr,
            orr=orr,
            n_total=len(all_ids),
            n_excluded=n_excluded,
        ))

    return rows


# ── Per-sample variance ────────────────────────────────────────────────────

def _compute_per_sample_variance(value_map: dict) -> dict:
    result = {}
    for sid, ep_vals in value_map.items():
        values = [v for v in ep_vals.values() if v is not None]
        result[sid] = statistics.variance(values) if len(values) >= 2 else None
    return result


# ── Text output ────────────────────────────────────────────────────────────

def _fmt_pct(val) -> str:
    return f"{val * 100:.1f}%" if val is not None else "—"


def _print_convergence_table(rows: list, lang: str, n_samples: int, n_epochs: int) -> None:
    print(f"{'─' * 60}")
    print(f"Convergence analysis: {lang}  ({n_samples} samples, {n_epochs} epoch(s))")
    print()

    if n_epochs == 1:
        print("  Only 1 epoch — no convergence curve. Run with epochs > 1.")
        print()

    header = f"  {'k':>7}  {'Accuracy':>10}  {'ASR (unsafe)':>14}  {'ORR (safe)':>12}  {'Δ-accuracy':>12}"
    sep    = f"  {'─'*7}  {'─'*10}  {'─'*14}  {'─'*12}  {'─'*12}"
    print(header)
    print(sep)

    prev_acc = None
    for row in rows:
        acc = row["accuracy"]
        if prev_acc is not None and acc is not None and prev_acc is not None:
            delta_val = (acc - prev_acc) * 100
            delta = f"{delta_val:+.1f}pp"
        else:
            delta = "—"

        excluded_note = f"  ({row['n_excluded']} excluded)" if row["n_excluded"] else ""
        print(
            f"  {row['k']:>7}  {_fmt_pct(acc):>10}  {_fmt_pct(row['asr']):>14}"
            f"  {_fmt_pct(row['orr']):>12}  {delta:>12}{excluded_note}"
        )
        prev_acc = acc

    print()


def _print_variance_summary(variance_by_sid: dict, label_map: dict, top_n: int = 10) -> None:
    ranked = sorted(
        [(sid, v) for sid, v in variance_by_sid.items() if v is not None],
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    if not ranked:
        return

    print(f"{'─' * 60}")
    print(f"Top-{min(top_n, len(ranked))} highest-variance samples (most borderline)")
    print()
    print(f"  {'Sample ID':<35}  {'Label':<10}  {'Variance':>8}")
    print(f"  {'─'*35}  {'─'*10}  {'─'*8}")
    for sid, var in ranked:
        label = label_map.get(sid, "unknown")
        print(f"  {sid:<35}  {label:<10}  {var:>8.4f}")
    print()


# ── Plot ───────────────────────────────────────────────────────────────────

def _plot_convergence(rows: list, lang: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [plot] matplotlib/numpy not installed — skipping visualisation.")
        return

    if len(rows) < 2:
        print("  [plot] Need at least 2 epochs to plot a convergence curve.")
        return

    ks       = [r["k"] for r in rows]
    accs     = [r["accuracy"] * 100 if r["accuracy"] is not None else np.nan for r in rows]
    asrs     = [r["asr"]      * 100 if r["asr"]      is not None else np.nan for r in rows]
    orrs     = [r["orr"]      * 100 if r["orr"]      is not None else np.nan for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # ── Subplot 1: running metrics ──────────────────────────────────────
    ax1.plot(ks, accs, "b-o",  label="Accuracy",        linewidth=1.8, markersize=5)
    ax1.plot(ks, asrs, "r--x", label="ASR (unsafe)",    linewidth=1.5, markersize=5)
    ax1.plot(ks, orrs, ":-s",  label="ORR (safe)",      linewidth=1.5, markersize=5, color="orange")

    final_acc = next((a for a in reversed(accs) if not np.isnan(a)), None)
    if final_acc is not None:
        ax1.axhline(final_acc, color="grey", linestyle="--", linewidth=0.8, alpha=0.6,
                    label=f"Final accuracy ({final_acc:.1f}%)")

    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Metric (%)", fontsize=11)
    ax1.set_title(f"Convergence — {lang}  ({rows[0]['n_total']} samples, {len(rows)} epochs)",
                  fontsize=13, pad=10)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # ── Subplot 2: delta accuracy ───────────────────────────────────────
    deltas = []
    delta_ks = []
    for i in range(1, len(rows)):
        a_prev = accs[i - 1]
        a_curr = accs[i]
        if not (np.isnan(a_prev) or np.isnan(a_curr)):
            deltas.append(abs(a_curr - a_prev))
            delta_ks.append(ks[i])

    bar_colors = [
        "#4CAF50" if d <= 1 else "#FF9800" if d <= 2 else "#F44336"
        for d in deltas
    ]
    ax2.bar(delta_ks, deltas, color=bar_colors, edgecolor="white", linewidth=0.6)
    ax2.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7,
                label="1pp (stable)")
    ax2.set_ylabel("|Δ accuracy| (pp)", fontsize=11)
    ax2.set_xlabel("Epoch count k", fontsize=11)
    ax2.set_title("|Δ accuracy| per additional epoch", fontsize=11, pad=8)
    ax2.set_xticks(ks)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plt.show()


# ── Per-file orchestration ─────────────────────────────────────────────────

def analyze_log(log_path: Path, plot: bool = False) -> None:
    print(f"  Loading {log_path.name} ...", end=" ", flush=True)
    try:
        log = read_eval_log(str(log_path))
    except Exception as e:
        print(f"ERROR: {e}")
        return

    lang = (log.eval.task_args or {}).get("language", "unknown")
    print(f"lang={lang}")

    epoch_map, label_map, value_map, n_epochs, parse_error_ids = _load_epoch_map(log)

    if n_epochs == 0:
        print("  WARNING: no samples found.\n")
        return

    if n_epochs == 1 and all(
        list(ep_dict.keys()) == [1] for ep_dict in epoch_map.values()
        if ep_dict
    ):
        # All samples are epoch=1 — probably a multi-file log
        print(
            "  WARNING: all samples have epoch=1. This may be a multi-file log. "
            "For convergence analysis, run with epochs > 1. "
            "For cross-file consistency, use analyze_determinism.py instead."
        )

    if parse_error_ids:
        print(f"  WARNING: {len(parse_error_ids)} PARSE_ERROR(s) found — excluded from accuracy.")

    n_samples = len(label_map)
    rows = _compute_running_metrics(epoch_map, label_map, n_epochs)
    _print_convergence_table(rows, lang, n_samples, n_epochs)

    variance_by_sid = _compute_per_sample_variance(value_map)
    _print_variance_summary(variance_by_sid, label_map)

    if plot:
        _plot_convergence(rows, lang)


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convergence analysis: does accuracy stabilize as epoch count increases?"
    )
    parser.add_argument("--logs", nargs="+", type=Path,
                        help="Specific .eval files (default: last --n logs)")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of most-recent logs to analyse (default: 5)")
    parser.add_argument("--plot", action="store_true",
                        help="Show convergence line charts (requires matplotlib)")
    args = parser.parse_args()

    if args.logs:
        log_paths = list(args.logs)
    else:
        all_logs = sorted(LOGS_DIR.glob("*.eval"))
        log_paths = all_logs[-args.n:]

    if not log_paths:
        print("No .eval files found.")
        return

    print(f"Analysing {len(log_paths)} log(s):\n")
    for path in log_paths:
        analyze_log(path, plot=args.plot)


if __name__ == "__main__":
    main()
