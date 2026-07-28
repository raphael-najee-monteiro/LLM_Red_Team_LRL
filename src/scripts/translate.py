#!/usr/bin/env python3
"""
Produce language-specific sample files from the English samples made by prepare_data.py.

de, ta  — Google Translate every prompt from the English sample files.

ar      — LinguaSafe adversarial prompts are looked up by ID in linguasafe_ar.jsonl
          (linguasafe_en.jsonl[i] is already translated as linguasafe_ar.jsonl[i]).
          XSTest safe prompts (English-only dataset) are Google Translated.

ch      — Swiss German (Zürich dialect) via the configured OpenAI-compatible API
          (Google Translate does not support Swiss German dialect).

Reads:   data/{subset}/en.jsonl          (produced by prepare_data.py)
         data/linguasafe_ar.jsonl        (for Arabic LinguaSafe lookup)

Writes:  data/{subset}/{lang}.jsonl

Only `input` is translated. `target` stays in English (read by the grader, not
the evaluated model).

Usage:
    pip install deep-translator openai
    python src/scripts/translate.py                        # all langs, all subsets
    python src/scripts/translate.py --lang ar              # Arabic only
    python src/scripts/translate.py --lang de ta --subset 10_sample
    python src/scripts/translate.py --lang ch --subset 100_sample
"""

import argparse
import json
import time
from pathlib import Path

from deep_translator import GoogleTranslator
from deep_translator.exceptions import RequestError, TooManyRequests
from dotenv import load_dotenv

load_dotenv()

# ── config ───────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_SUBSETS = ["10_sample", "50_sample", "100_sample", "300_sample"]
DEFAULT_LANGS = ["de", "ar", "ta", "ch"]

# ch uses LLM translation (Google Translate doesn't support Swiss German dialect)
GOOGLE_LANG: dict[str, str] = {"de": "de", "ar": "ar", "ta": "ta"}
LANG_NAME: dict[str, str] = {
    "de": "German (Hochdeutsch)",
    "ar": "Arabic",
    "ta": "Tamil",
    "ch": "Swiss German (Zürich dialect)",
}

REQUEST_DELAY = 0.5  # seconds between Google Translate requests

CH_SYSTEM_PROMPT = """\
You are a Swiss German (Zürich dialect / Züritüütsch) translator.
Translate the given English text into Swiss German Zürich dialect.

Rules:
- Use authentic Zürich dialect: 'isch' (ist), 'nöd' (nicht), 'chönd' (können),
  'händ' (haben), 'gah' (gehen), 'säge' (sagen), 'öpper' (jemand), 'öppis' (etwas).
- Keep proper nouns, technical terms, and placeholders like [name] or [company] unchanged.
- Output ONLY the translated text — no explanation, no quotes, no extra text.\
"""

# LLM used for Swiss German dialect translation. Google Translate does not support
# Swiss German, so an OpenAI-compatible model served via Ollama is used instead.
# Change this to any model available on your Ollama instance.
CH_MODEL = "gemma3:4b"


# ── helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def build_ar_lookup() -> dict[str, dict]:
    """
    Build {numeric_id: entry} from linguasafe_ar.jsonl.
    E.g. entry with id '42_ar' is stored under key '42'.
    """
    path = DATA_DIR / "linguasafe_ar.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run extract_linguasafe_ar.py first"
        )
    return {
        e["id"].rsplit("_", 1)[0]: e
        for e in load_jsonl(path)
    }


def llm_translate_ch(text: str, retries: int = 3) -> tuple[str, bool]:
    """Translate to Swiss German Zürich dialect via the configured OpenAI-compatible API."""
    from openai import OpenAI
    client = OpenAI()
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=CH_MODEL,
                messages=[
                    {"role": "system", "content": CH_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                seed=42,
                max_tokens=500,
            )
            result = response.choices[0].message.content.strip()
            return result, False
        except Exception as e:
            print(f"    [ch translate error attempt {attempt + 1}] {e}")
            if attempt == retries - 1:
                return text, True
            time.sleep(2 ** attempt)
    return text, True


def google_translate(text: str, target_lang: str, retries: int = 4) -> tuple[str, bool]:
    """Returns (translated_text, fell_back)."""
    for attempt in range(retries):
        try:
            result = GoogleTranslator(source="en", target=target_lang).translate(text)
            return (result or text, result is None or result == text)
        except TooManyRequests:
            wait = 2 ** (attempt + 1)
            print(f"    [rate limit] waiting {wait}s...")
            time.sleep(wait)
        except RequestError as e:
            print(f"    [request error attempt {attempt + 1}] {e}")
            if attempt == retries - 1:
                return text, True
            time.sleep(1)
        except Exception as e:
            print(f"    [unexpected error] {e}")
            return text, True
    return text, True


def translate_entry(e: dict, lang: str, ar_lookup: dict | None) -> tuple[dict, bool]:
    """
    Return (translated_entry, fell_back).
    For Arabic LinguaSafe entries: ID lookup (no API call).
    For Swiss German (ch): LLM-based translation.
    For everything else: Google Translate.
    """
    numeric_id = e["id"].rsplit("_en", 1)[0]   # "42_en" → "42"
    new_id = f"{numeric_id}_{lang}"

    if lang == "ar" and e["metadata"].get("dataset") == "linguasafe":
        ar_entry = ar_lookup.get(numeric_id)
        if ar_entry:
            return (
                {**e, "input": ar_entry["input"], "id": new_id,
                 "metadata": {**e["metadata"], "lang": lang}},
                False,
            )
        # ID not in Arabic file (extra English-only entries) → fall back to Google
        print(f"    [no AR lookup for {numeric_id}] falling back to Google Translate")

    if lang == "ch":
        translated, fell_back = llm_translate_ch(e["input"])
    else:
        translated, fell_back = google_translate(e["input"], GOOGLE_LANG[lang])

    meta = {**e["metadata"], "lang": lang}
    if fell_back:
        meta["translation_fallback"] = "true"
    return {**e, "input": translated, "id": new_id, "metadata": meta}, fell_back


def process_subset(subset: str, lang: str, ar_lookup: dict | None) -> int:
    """Translate one subset file for one language. Returns fallback count."""
    src = DATA_DIR / subset / "en.jsonl"
    if not src.exists():
        print(f"  SKIP {subset}/en.jsonl not found — run prepare_data.py first")
        return 0

    entries = load_jsonl(src)
    result = []
    fallbacks = 0

    for i, e in enumerate(entries, 1):
        translated, fell_back = translate_entry(e, lang, ar_lookup)
        fallbacks += fell_back
        result.append(translated)

        if lang == "ar" and e["metadata"].get("dataset") == "linguasafe" and not fell_back:
            method = "lookup"
        elif lang == "ch":
            method = "llm"
        else:
            method = "translate"
        status = " [FALLBACK]" if fell_back else ""
        print(f"  [{i:>3}/{len(entries)}] ({method}){status}")
        print(f"    EN: {e['input'][:60].replace(chr(10), ' ')!r}")
        print(f"    {lang.upper()}: {translated['input'][:60].replace(chr(10), ' ')!r}")

        if method == "translate":
            time.sleep(REQUEST_DELAY)

    out = DATA_DIR / subset / f"{lang}.jsonl"
    write_jsonl(out, result)
    print(f"\n  Saved {len(result)} entries -> {out.relative_to(DATA_DIR.parent.parent)}"
          f"  (fallbacks: {fallbacks})")
    return fallbacks


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce language sample files from English samples"
    )
    parser.add_argument("--lang", nargs="+", default=DEFAULT_LANGS,
                        choices=list(LANG_NAME.keys()))
    parser.add_argument("--subset", nargs="+", default=DEFAULT_SUBSETS,
                        choices=DEFAULT_SUBSETS, metavar="SUBSET")
    args = parser.parse_args()

    ar_lookup = build_ar_lookup() if "ar" in args.lang else None

    total_fallbacks = 0

    for lang in args.lang:
        print(f"\n{'=' * 60}")
        print(f"  {LANG_NAME[lang]} ({lang})")
        print(f"{'=' * 60}")

        for subset in args.subset:
            print(f"\n-- {subset} --")
            total_fallbacks += process_subset(subset, lang, ar_lookup)

    if total_fallbacks:
        print(f"\nWARNING: {total_fallbacks} prompt(s) kept English original.")
        print("Flagged with metadata.translation_fallback = true.")
    else:
        print("\nAll translations completed successfully.")


if __name__ == "__main__":
    main()
