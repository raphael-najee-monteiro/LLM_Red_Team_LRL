#!/usr/bin/env python3
"""
Analyze grader determinism across repeated runs of the same experiment.

Supports two modes (auto-detected):

  Multi-file mode  — the last N logs are K runs × L languages, each run as a
                     separate .eval file (e.g. 3 runs × 5 languages = 15 files).
                     Samples are compared across files that share the same language.

  Epoch mode       — a single .eval file per language was produced with epochs=K,
                     so each sample appears K times (one per epoch) inside that file.
                     Samples are compared across epochs within the same file.

Usage:
    python src/scripts/analyze_determinism.py               # last 15 logs (multi-file)
    python src/scripts/analyze_determinism.py --n 5         # last 5 logs  (epoch mode: 1 per lang)
    python src/scripts/analyze_determinism.py --logs src/logs/foo.eval ...
"""

import argparse
import collections
from pathlib import Path

from inspect_ai.log import read_eval_log, EvalLog

LOGS_DIR = Path(__file__).parent.parent / "logs"


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_answer(sample) -> str:
    score_obj = (sample.scores or {}).get("yes_no_grader")
    if score_obj is None:
        score_obj = next(iter((sample.scores or {}).values()), None)
    return score_obj.answer if score_obj else "MISSING"


def _is_epoch_log(log: EvalLog) -> bool:
    """Return True if this log has more than one epoch (epochs > 1 in Task)."""
    return any(getattr(s, "epoch", 1) > 1 for s in (log.samples or []))


# ── Per-language analysis (shared by both modes) ───────────────────────────

def _report_language(lang: str, runs: list[list[tuple[str, str]]]) -> tuple[int, int]:
    """
    runs: list of runs, each run is a list of (sample_id, answer) pairs.
    Returns (n_total, n_consistent).
    """
    print(f"{'─'*60}")
    print(f"Language: {lang}  ({len(runs)} run(s) / epoch(s))")

    if len(runs) < 2:
        print("  Only 1 run — nothing to compare.\n")
        return 0, 0

    # Map sample_id -> list of answers (one per run/epoch)
    all_ids: set[str] = set()
    for run in runs:
        all_ids.update(sid for sid, _ in run)

    votes_by_id: dict[str, list[str]] = {}
    for sid in sorted(all_ids):
        votes_by_id[sid] = []
        for run in runs:
            run_dict = dict(run)
            votes_by_id[sid].append(run_dict.get(sid, "MISSING"))

    consistent = sum(1 for v in votes_by_id.values() if len(set(v)) == 1)
    inconsistent_rows = [(sid, v) for sid, v in votes_by_id.items() if len(set(v)) > 1]
    n = len(all_ids)
    pct = consistent / n * 100 if n else 0

    print(f"  Samples      : {n}")
    print(f"  Consistent   : {consistent}/{n}  ({pct:.1f}%)")
    print(f"  Inconsistent : {len(inconsistent_rows)}/{n}  ({100-pct:.1f}%)")

    if inconsistent_rows:
        n_runs = len(runs)
        header_cols = "  ".join(f"{'Run '+str(i+1):<6}" for i in range(n_runs))
        sep_cols    = "  ".join([f"{'------':<6}"] * n_runs)
        print(f"\n  Inconsistent samples:")
        print(f"  {'Sample ID':<35}{header_cols}")
        print(f"  {'-'*35}{sep_cols}")
        for sid, votes in inconsistent_rows:
            row = f"  {sid:<35}" + "  ".join(f"{v:<6}" for v in votes)
            print(row)

    print()
    return n, consistent


# ── Multi-file mode ────────────────────────────────────────────────────────

def analyze_multifile(log_paths: list[Path], plot: bool = False) -> None:
    print(f"Mode: multi-file  ({len(log_paths)} log files)\n")

    # lang -> list of [(sample_id, answer), ...]
    by_lang: dict[str, list[list[tuple[str, str]]]] = collections.defaultdict(list)
    meta_by_id: dict[str, dict] = {}

    for path in log_paths:
        print(f"  Loading {path.name} ...", end=" ", flush=True)
        try:
            log = read_eval_log(str(path))
            lang = (log.eval.task_args or {}).get("language", "unknown")
            run = []
            for s in (log.samples or []):
                if getattr(s, "epoch", 1) == 1:
                    run.append((s.id, _get_answer(s)))
                if s.id not in meta_by_id:
                    meta_by_id[s.id] = s.metadata or {}
            by_lang[lang].append(run)
            print(f"lang={lang}, {len(run)} samples")
        except Exception as e:
            print(f"ERROR: {e}")

    print()
    _summarize(by_lang, meta_by_id=meta_by_id, plot=plot)


# ── Epoch mode ─────────────────────────────────────────────────────────────

def analyze_epochs(log_paths: list[Path], plot: bool = False) -> None:
    print(f"Mode: epoch  ({len(log_paths)} log file(s))\n")

    # lang -> list of [(sample_id, answer), ...] — one list per epoch
    by_lang: dict[str, list[list[tuple[str, str]]]] = collections.defaultdict(list)
    meta_by_id: dict[str, dict] = {}

    for path in log_paths:
        print(f"  Loading {path.name} ...", end=" ", flush=True)
        try:
            log = read_eval_log(str(path))
            lang = (log.eval.task_args or {}).get("language", "unknown")

            # Group samples by epoch number
            epochs: dict[int, list[tuple[str, str]]] = collections.defaultdict(list)
            for s in (log.samples or []):
                ep = getattr(s, "epoch", 1)
                epochs[ep].append((s.id, _get_answer(s)))
                if s.id not in meta_by_id:
                    meta_by_id[s.id] = s.metadata or {}

            n_epochs = len(epochs)
            n_samples = len(epochs[min(epochs)]) if epochs else 0
            print(f"lang={lang}, {n_epochs} epoch(s), {n_samples} samples each")

            for ep in sorted(epochs):
                by_lang[lang].append(epochs[ep])
        except Exception as e:
            print(f"ERROR: {e}")

    print()
    _summarize(by_lang, meta_by_id=meta_by_id, plot=plot)


# ── Breakdown helpers ───────────────────────────────────────────────────────

def _compute_consistency_by_sid(
    by_lang: dict[str, list[list[tuple[str, str]]]],
) -> dict[str, bool]:
    """Return {sample_id: is_consistent} for every sample with >= 2 runs."""
    result: dict[str, bool] = {}
    for runs in by_lang.values():
        if len(runs) < 2:
            continue
        all_ids = set(sid for run in runs for sid, _ in run)
        for sid in all_ids:
            votes = [dict(run).get(sid, "MISSING") for run in runs]
            result[sid] = len(set(votes)) == 1
    return result


def _report_breakdown(
    field: str,
    label: str,
    consistency_by_sid: dict[str, bool],
    meta_by_id: dict[str, dict],
) -> dict[str, tuple[int, int]]:
    """Print consistency table grouped by a metadata field. Returns {group: (n, consistent)}."""
    by_group: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for sid, is_consistent in consistency_by_sid.items():
        group = (meta_by_id.get(sid) or {}).get(field, "unknown") or "unknown"
        by_group[group][0] += 1
        if is_consistent:
            by_group[group][1] += 1

    if not by_group:
        return {}

    col1 = max(len(g) for g in by_group) + 2
    print(f"\n{'─'*60}")
    print(f"Breakdown by {label}")
    print(f"{'─'*60}")
    print(f"  {'Group':<{col1}}{'Samples':>10}  {'Consistent':>10}  {'%':>6}")
    print(f"  {'─'*col1}{'─'*10}  {'─'*10}  {'─'*6}")
    result: dict[str, tuple[int, int]] = {}
    for group in sorted(by_group):
        n, c = by_group[group]
        pct = c / n * 100 if n else 0
        print(f"  {group:<{col1}}{n:>10}  {c:>10}  {pct:>5.1f}%")
        result[group] = (n, c)
    return result


# ── Shared summary ─────────────────────────────────────────────────────────

def _plot_results(
    by_lang: dict[str, list[list[tuple[str, str]]]],
    breakdowns: dict[str, dict[str, tuple[int, int]]] | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n  [plot] matplotlib/numpy not installed — skipping visualisation.")
        return

    langs = [l for l, runs in sorted(by_lang.items()) if len(runs) >= 2]
    if not langs:
        return

    # ── 1. Bar chart: consistency % per language ────────────────────────
    consistencies: list[float] = []
    for lang in langs:
        runs = by_lang[lang]
        all_ids = set(sid for run in runs for sid, _ in run)
        n = len(all_ids)
        if n == 0:
            consistencies.append(0.0)
            continue
        consistent = sum(
            1 for sid in all_ids
            if len({dict(run).get(sid, "MISSING") for run in runs}) == 1
        )
        consistencies.append(consistent / n * 100)

    bar_colors = [
        "#4CAF50" if c >= 90 else "#FF9800" if c >= 70 else "#F44336"
        for c in consistencies
    ]

    fig_bar, ax_bar = plt.subplots(figsize=(max(6, len(langs) * 1.5 + 1), 5))
    bars = ax_bar.bar(langs, consistencies, color=bar_colors, edgecolor="white", linewidth=0.8)
    ax_bar.set_ylim(0, 115)
    ax_bar.set_ylabel("Consistency (%)", fontsize=11)
    ax_bar.set_title("Grader Determinism — Consistency per Language", fontsize=13, pad=12)
    ax_bar.axhline(100, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    for bar, pct in zip(bars, consistencies):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax_bar.tick_params(axis="x", labelsize=10)
    fig_bar.tight_layout()

    # ── 2. Heatmap: samples × runs per language ─────────────────────────
    all_answers: set[str] = set()
    for lang in langs:
        for run in by_lang[lang]:
            all_answers.update(ans for _, ans in run)
    answer_list = sorted(all_answers)
    answer_to_int = {a: i for i, a in enumerate(answer_list)}

    n_cols = min(3, len(langs))
    n_rows = (len(langs) + n_cols - 1) // n_cols
    fig_heat, axes_grid = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 5, max(3, n_rows * 4)),
        squeeze=False,
    )
    axes_flat = axes_grid.flatten()

    cmap = plt.colormaps.get_cmap("RdYlGn").resampled(max(len(answer_list), 2))

    for ax, lang in zip(axes_flat, langs):
        runs = by_lang[lang]
        all_ids = sorted(set(sid for run in runs for sid, _ in run))
        matrix = np.full((len(all_ids), len(runs)), np.nan)
        for j, run in enumerate(runs):
            run_dict = dict(run)
            for i, sid in enumerate(all_ids):
                ans = run_dict.get(sid, "MISSING")
                matrix[i, j] = answer_to_int.get(ans, -1)

        im = ax.imshow(
            matrix, aspect="auto", cmap=cmap,
            vmin=-0.5, vmax=len(answer_list) - 0.5,
            interpolation="nearest",
        )
        ax.set_title(lang, fontsize=11, pad=6)
        ax.set_xlabel("Run / Epoch", fontsize=9)
        ax.set_ylabel("Sample index", fontsize=9)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels([f"R{i+1}" for i in range(len(runs))], fontsize=8)
        ax.set_yticks(range(len(all_ids)))
        ax.set_yticklabels(range(1, len(all_ids) + 1), fontsize=6)

        cbar = fig_heat.colorbar(im, ax=ax, ticks=range(len(answer_list)), pad=0.02)
        cbar.ax.set_yticklabels(answer_list, fontsize=7)

    for ax in axes_flat[len(langs):]:
        ax.set_visible(False)

    fig_heat.suptitle("Answer Heatmap — Samples × Runs/Epochs", fontsize=13, y=1.01)
    fig_heat.tight_layout()

    # ── 3. Breakdown charts (category + origin) ──────────────────────────
    if breakdowns:
        def _breakdown_chart(ax: "plt.Axes", data: dict[str, tuple[int, int]], title: str) -> None:
            groups = sorted(data)
            pcts = [data[g][1] / data[g][0] * 100 if data[g][0] else 0 for g in groups]
            colors = ["#4CAF50" if p >= 90 else "#FF9800" if p >= 70 else "#F44336" for p in pcts]
            bars = ax.barh(groups, pcts, color=colors, edgecolor="white", linewidth=0.6)
            ax.set_xlim(0, 115)
            ax.set_xlabel("Consistency (%)", fontsize=9)
            ax.set_title(title, fontsize=11, pad=8)
            ax.axvline(100, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
            for bar, pct in zip(bars, pcts):
                ax.text(pct + 1.5, bar.get_y() + bar.get_height() / 2,
                        f"{pct:.1f}%", va="center", fontsize=8, fontweight="bold")
            ax.tick_params(axis="y", labelsize=8)

        n_bd = len(breakdowns)
        fig_bd, bd_axes = plt.subplots(1, n_bd, figsize=(n_bd * 7, max(3, max(len(v) for v in breakdowns.values()) * 0.45 + 1.5)))
        if n_bd == 1:
            bd_axes = [bd_axes]
        titles = {"category": "Consistency by Category", "origin": "Consistency by Dataset Origin"}
        for ax, (key, data) in zip(bd_axes, breakdowns.items()):
            if data:
                _breakdown_chart(ax, data, titles.get(key, key))
        fig_bd.suptitle("Determinism Breakdowns", fontsize=13)
        fig_bd.tight_layout()

    plt.show()


def _summarize(
    by_lang: dict[str, list[list[tuple[str, str]]]],
    meta_by_id: dict[str, dict] | None = None,
    plot: bool = False,
) -> None:
    total_samples = 0
    total_consistent = 0

    for lang in sorted(by_lang):
        n, c = _report_language(lang, by_lang[lang])
        total_samples += n
        total_consistent += c

    langs_compared = [l for l, runs in by_lang.items() if len(runs) >= 2]
    print(f"{'═'*60}")
    print("OVERALL DETERMINISM SUMMARY")
    print(f"{'═'*60}")
    if total_samples:
        pct = total_consistent / total_samples * 100
        print(f"  Languages analysed : {', '.join(sorted(langs_compared))}")
        print(f"  Total sample slots : {total_samples}")
        print(f"  Consistent         : {total_consistent}  ({pct:.1f}%)")
        print(f"  Inconsistent       : {total_samples - total_consistent}  ({100-pct:.1f}%)")

    breakdowns: dict[str, dict[str, tuple[int, int]]] = {}
    if meta_by_id:
        consistency_by_sid = _compute_consistency_by_sid(by_lang)
        breakdowns["category"] = _report_breakdown(
            "type", 'category  (metadata["type"])', consistency_by_sid, meta_by_id
        )
        breakdowns["origin"] = _report_breakdown(
            "label", 'dataset origin  (metadata["label"])', consistency_by_sid, meta_by_id
        )

    if plot:
        _plot_results(by_lang, breakdowns=breakdowns or None)


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Determinism analysis across repeated runs")
    parser.add_argument("--logs", nargs="+", type=Path,
                        help="Specific .eval files (default: last --n logs)")
    parser.add_argument("--n", type=int, default=15,
                        help="Number of most-recent logs to analyse (default: 15)")
    parser.add_argument("--mode", choices=["auto", "multifile", "epoch"], default="auto",
                        help="Force a mode (default: auto-detect)")
    parser.add_argument("--plot", action="store_true",
                        help="Show bar chart and heatmap after analysis (requires matplotlib)")
    args = parser.parse_args()

    if args.logs:
        log_paths = sorted(args.logs)
    else:
        all_logs = sorted(LOGS_DIR.glob("*.eval"))
        log_paths = all_logs[-args.n:]

    if not log_paths:
        print("No .eval files found.")
        return

    print(f"Analysing {len(log_paths)} log(s):\n")

    # Auto-detect mode: peek at the first log
    mode = args.mode
    if mode == "auto":
        try:
            first = read_eval_log(str(log_paths[0]))
            mode = "epoch" if _is_epoch_log(first) else "multifile"
        except Exception:
            mode = "multifile"

    if mode == "epoch":
        analyze_epochs(log_paths, plot=args.plot)
    else:
        analyze_multifile(log_paths, plot=args.plot)


if __name__ == "__main__":
    main()