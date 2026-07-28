from dotenv import load_dotenv
load_dotenv()


# Which dataset to evaluate.
# Options: "all", "10_sample", "50_sample", "100_sample", "300_sample"
# The published experiments used "50_sample" (35 adversarial + 15 safe prompts).
DATASET = "50_sample"

# Grader model — independent judge that scores model responses.
# The published experiments used Mistral 7B.
GRADER_MODEL = "openai/mistral"

# Target models to evaluate.
MODELS = [
    "openai/gemma3:270m",
    "openai/deepseek-r1:1.5b",
    "openai/MichelRosselli/apertus:8b-instruct-2509-q4_k_m",
]

# Languages to evaluate across (ISO codes used as dataset file names).
LANGUAGES = ["en", "de", "ar", "ch", "ta"]

# ── Reproducibility settings ──────────────────────────────────────────────────
# Applied to both the evaluated model and the grader model.
# temperature=0 → greedy decoding (fully deterministic given a fixed seed).
# Keep these fixed across all runs so results are comparable.
N_RUNS = 15

CONFIG = {
    "seed":        42,
    "temperature": 0.0,
    "top_p":       1.0,    # nucleus sampling; irrelevant at temp=0 but explicit
    "max_tokens":  500,    # evaluated model response cap
}

GRADER_CONFIG = {
    "seed":        42,
    "temperature": 0.0,
    "top_p":       1.0,
    "max_tokens":  300,    # grader only needs space for the structured fields
}
