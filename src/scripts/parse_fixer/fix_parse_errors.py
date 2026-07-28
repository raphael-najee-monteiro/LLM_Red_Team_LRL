#!/usr/bin/env python3
"""
Scan .eval log files for PARSE_ERROR samples, recover the grade from the
stored grader explanation using a tolerant regex, and patch the log in-place.

See README.md in this directory for full context.

Usage:
    python src/scripts/parse_fixer/fix_parse_errors.py              # all logs
    python src/scripts/parse_fixer/fix_parse_errors.py --dry-run    # report only
    python src/scripts/parse_fixer/fix_parse_errors.py --log src/logs/foo.eval
"""

import argparse
import json
import re
import shutil
import struct
import unicodedata
import zipfile
from pathlib import Path

try:
    import zstandard as _zstd
    _ZSTD_AVAILABLE = True
except ImportError:
    _ZSTD_AVAILABLE = False

_ZIP_METHOD_ZSTANDARD = 93


def _read_zip_entry(z: zipfile.ZipFile, log_path: Path, info: zipfile.ZipInfo) -> bytes:
    """Read a single ZIP entry, handling Zstandard compression (method 93)."""
    if info.compress_type != _ZIP_METHOD_ZSTANDARD:
        return z.read(info.filename)
    if not _ZSTD_AVAILABLE:
        raise RuntimeError("pip install zstandard  (InspectAI .eval files use Zstandard compression)")
    with open(log_path, "rb") as f:
        f.seek(info.header_offset)
        local_header = f.read(30)
        fname_len = struct.unpack_from("<H", local_header, 26)[0]
        extra_len = struct.unpack_from("<H", local_header, 28)[0]
        f.seek(info.header_offset + 30 + fname_len + extra_len)
        compressed = f.read(info.compress_size)
    dctx = _zstd.ZstdDecompressor()
    with dctx.stream_reader(compressed) as reader:
        return reader.read()

LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
BACKUP_DIR = LOGS_DIR / "backups"

# Tolerant recovery regex: handles **GRADE:** YES, *GRADE*: YES, GRADE: **YES**, etc.
# No ^ anchor needed: "GRADE CONFIDENCE" lines won't match RECOVERY_GRADE_RE because
# the regex requires a colon immediately after GRADE (plus optional asterisks/spaces),
# and GRADE CONFIDENCE has the word CONFIDENCE between GRADE and the colon.
RECOVERY_GRADE_RE = re.compile(
    r"(?i)\*{0,2}GRADE\*{0,2}\s*:\s*\*{0,2}\s*(YES|NO)"
)
RECOVERY_CONF_RE = re.compile(
    r"(?i)\*{0,2}GRADE CONFIDENCE\*{0,2}\s*:\s*\*{0,2}\s*([0-9]*\.?[0-9]+)"
)


def try_recover(explanation: str) -> tuple[str | None, float | None]:
    text = unicodedata.normalize("NFKC", explanation)
    grade_m = RECOVERY_GRADE_RE.search(text)
    conf_m = RECOVERY_CONF_RE.search(text)
    grade = grade_m.group(1).upper() if grade_m else None
    confidence = float(conf_m.group(1)) if conf_m else None
    return grade, confidence


def fix_log(log_path: Path, dry_run: bool) -> tuple[int, int]:
    """Returns (fixed_count, unrecoverable_count)."""
    fixed = 0
    unrecoverable = 0

    with zipfile.ZipFile(log_path, "r") as z:
        infos = {info.filename: info for info in z.infolist()}
        names = list(infos)
        contents: dict[str, bytes] = {name: _read_zip_entry(z, log_path, infos[name]) for name in names}

    # Load all summary/reduction files we need to keep in sync with sample changes.
    # _journal/summaries/<epoch>.json — one array per epoch (InspectAI internal)
    journal_summaries: dict[str, list] = {}
    for name in names:
        if name.startswith("_journal/summaries/") and name.endswith(".json"):
            epoch = Path(name).stem
            journal_summaries[epoch] = json.loads(contents[name])

    # summaries.json — flat list of ALL samples across ALL epochs (viewer reads this)
    flat_summaries: list = json.loads(contents["summaries.json"]) if "summaries.json" in contents else []

    # reductions.json — per-sample mean values across epochs (viewer shows these)
    reductions: list = json.loads(contents["reductions.json"]) if "reductions.json" in contents else []

    # Track which sample_ids were modified so we can recompute their reduction values.
    modified_sample_ids: set[str] = set()
    modified = False

    for name in names:
        if not (name.startswith("samples/") and name.endswith(".json")):
            continue

        data = json.loads(contents[name])
        score = (data.get("scores") or {}).get("yes_no_grader")
        if not score or score.get("answer") != "PARSE_ERROR":
            continue

        explanation = score.get("explanation", "")
        grade, confidence = try_recover(explanation)

        if grade is None:
            print(f"  UNRECOVERABLE  {name}")
            print(f"    explanation: {repr(explanation[:120])}")
            unrecoverable += 1
            continue

        print(f"  RECOVERED  {name}  ->  {grade}" + (f"  (conf={confidence})" if confidence is not None else ""))
        fixed += 1

        if not dry_run:
            new_value = 1.0 if grade == "YES" else 0.0

            # 1. Patch the sample file.
            score["value"] = new_value
            score["answer"] = grade
            if confidence is not None:
                score.setdefault("metadata", {})["grade_confidence"] = confidence
            contents[name] = json.dumps(data, ensure_ascii=False).encode("utf-8")

            # Parse "samples/<sample_id>_epoch_<N>.json"
            stem = Path(name).stem  # e.g. "3098_ta-LK_epoch_3"
            parts = stem.rsplit("_epoch_", 1)
            if len(parts) != 2:
                modified = True
                continue
            sample_id, epoch = parts[0], parts[1]
            modified_sample_ids.add(sample_id)

            def _patch_score(s: dict) -> None:
                s["value"] = new_value
                s["answer"] = grade
                if confidence is not None:
                    s.setdefault("metadata", {})["grade_confidence"] = confidence

            # 2. Patch _journal/summaries/<epoch>.json
            if epoch in journal_summaries:
                for item in journal_summaries[epoch]:
                    if item.get("id") == sample_id:
                        s = (item.get("scores") or {}).get("yes_no_grader")
                        if s:
                            _patch_score(s)
                        break
                contents[f"_journal/summaries/{epoch}.json"] = json.dumps(
                    journal_summaries[epoch], ensure_ascii=False
                ).encode("utf-8")

            # 3. Patch summaries.json (flat list, match by id + epoch)
            epoch_int = int(epoch)
            for item in flat_summaries:
                if item.get("id") == sample_id and item.get("epoch") == epoch_int:
                    s = (item.get("scores") or {}).get("yes_no_grader")
                    if s:
                        _patch_score(s)
                    break

            modified = True

    if not dry_run and modified:
        # 4. Write updated summaries.json
        contents["summaries.json"] = json.dumps(flat_summaries, ensure_ascii=False).encode("utf-8")

        # 5. Recompute reductions for modified sample_ids.
        #    A reduction entry is the mean of each sample's value across all epochs.
        for reduction in reductions:
            if reduction.get("scorer") != "yes_no_grader":
                continue
            for red_sample in reduction.get("samples", []):
                sid = red_sample.get("sample_id")
                if sid not in modified_sample_ids:
                    continue
                # Collect all epoch values for this sample_id from the (patched) sample files.
                epoch_values = []
                for n in names:
                    if not (n.startswith("samples/") and n.endswith(".json")):
                        continue
                    stem = Path(n).stem
                    parts = stem.rsplit("_epoch_", 1)
                    if len(parts) == 2 and parts[0] == sid:
                        d = json.loads(contents[n])
                        s = (d.get("scores") or {}).get("yes_no_grader")
                        if s is not None:
                            epoch_values.append(float(s["value"]))
                if epoch_values:
                    red_sample["value"] = sum(epoch_values) / len(epoch_values)
        contents["reductions.json"] = json.dumps(reductions, ensure_ascii=False).encode("utf-8")

        # 6. Recompute accuracy/mean in header.json from the updated reductions.
        if "header.json" in contents:
            header = json.loads(contents["header.json"])
            for score_entry in (header.get("results") or {}).get("scores", []):
                scorer_name = score_entry.get("scorer") or score_entry.get("name")
                matching_reduction = next(
                    (r for r in reductions if r.get("scorer") == scorer_name), None
                )
                if matching_reduction is None:
                    continue
                values = [
                    s["value"] for s in matching_reduction.get("samples", [])
                    if "value" in s
                ]
                if not values:
                    continue
                new_mean = sum(values) / len(values)
                for metric in (score_entry.get("metrics") or {}).values():
                    if isinstance(metric, dict) and metric.get("name") in ("accuracy", "mean"):
                        metric["value"] = new_mean

                # Recompute scored_samples / unscored_samples.
                # scored  = unique sample_ids with at least 1 non-PARSE_ERROR epoch
                # unscored = unique sample_ids where ALL epochs are PARSE_ERROR
                sample_epoch_answers: dict[str, list[str]] = {}
                for n in names:
                    if not (n.startswith("samples/") and n.endswith(".json")):
                        continue
                    stem = Path(n).stem
                    parts = stem.rsplit("_epoch_", 1)
                    if len(parts) != 2:
                        continue
                    sid = parts[0]
                    d = json.loads(contents[n])
                    s = (d.get("scores") or {}).get(scorer_name)
                    answer = (s or {}).get("answer", "PARSE_ERROR")
                    sample_epoch_answers.setdefault(sid, []).append(answer)

                score_entry["scored_samples"] = sum(
                    1 for answers in sample_epoch_answers.values()
                    if any(a != "PARSE_ERROR" for a in answers)
                )
                score_entry["unscored_samples"] = sum(
                    1 for answers in sample_epoch_answers.values()
                    if all(a == "PARSE_ERROR" for a in answers)
                )

            contents["header.json"] = json.dumps(header, ensure_ascii=False).encode("utf-8")

    if modified:
        BACKUP_DIR.mkdir(exist_ok=True)
        backup = BACKUP_DIR / log_path.name
        if not backup.exists():
            shutil.copy2(log_path, backup)
            print(f"  Backup saved -> backups/{backup.name}")

        tmp = log_path.with_suffix(".eval.tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for name, data in contents.items():
                zout.writestr(name, data)
        tmp.replace(log_path)

    return fixed, unrecoverable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover PARSE_ERROR grades from stored grader explanations in .eval logs"
    )
    parser.add_argument(
        "--log", nargs="+", type=Path,
        help="Specific .eval file(s) to process (default: all in src/logs/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be fixed without writing any changes",
    )
    args = parser.parse_args()

    log_paths = args.log or sorted(LOGS_DIR.glob("*.eval"))
    if not log_paths:
        print(f"No .eval files found in {LOGS_DIR}")
        return

    if args.dry_run:
        print("DRY RUN — no files will be modified\n")

    total_fixed = 0
    total_unrecoverable = 0

    for path in log_paths:
        print(f"Scanning {path.name} ...")
        try:
            fixed, unrecoverable = fix_log(path, dry_run=args.dry_run)
        except Exception as e:
            print(f"  ERROR reading log: {e}")
            continue

        if fixed == 0 and unrecoverable == 0:
            print("  (no PARSE_ERRORs)")
        else:
            total_fixed += fixed
            total_unrecoverable += unrecoverable

    suffix = " (dry run — nothing written)" if args.dry_run else ""
    print(f"\nDone. Recovered: {total_fixed}  |  Unrecoverable: {total_unrecoverable}{suffix}")
    if total_unrecoverable:
        print("  Unrecoverable entries still have answer=PARSE_ERROR and value=0.0 in the logs.")
    if not args.dry_run and total_fixed:
        print("  Original files backed up in src/logs/backups/  |  Run export_results.py to refresh CSV.")


if __name__ == "__main__":
    main()
