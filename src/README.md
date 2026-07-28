# Running the Experiment

There are two ways to run the evaluation: locally (using Open WebUI as the model backend) or on Google Colab (using Ollama on a free T4 GPU).

---

## Option A — Google Colab (recommended for GPU runs)

The notebook `colab_eval.ipynb` handles the full setup automatically: it installs Ollama, pulls the models, writes `config.py`, and runs `eval.py`. No local setup needed beyond uploading the project to Google Drive.

> **Alternative:** `colab_eval_vsc.ipynb` skips Google Drive entirely. Open it in VS Code with the Colab extension, upload `eval.py` and `data/` directly to the Colab session, and download the resulting logs manually when the run finishes.

### 1. Upload the project to Google Drive

Copy the `src/` folder (containing `eval.py`, `config.py`, `data/`, etc.) into your Google Drive. Note the path, you'll need it in the next step.

### 2. Open the notebook in Colab

Open `colab_eval.ipynb` directly in Google Colab (right-click the file in Drive → *Open with → Google Colaboratory*).

### 3. Set the runtime to GPU

*Runtime → Change runtime type → Hardware accelerator → T4 GPU*

This is required before running any cells.

### 4. Set `PROJECT_PATH`

In the first code cell, set `PROJECT_PATH` to the folder you uploaded in step 1:

```python
PROJECT_PATH = '/content/drive/MyDrive/BA_experiments'  # adjust to your path
```

### 5. Configure the run

Edit the **Eval configuration** cell to choose models, languages, and dataset. This cell automatically overwrites `config.py` in your Drive folder. You do not need to edit `config.py` by hand.

```python
DATASET = "50_sample"   # "all" | "10_sample" | "50_sample" | "100_sample" | "300_sample"

MODELS = [
    "openai/gemma3:4b",
    # "openai/deepseek-r1:8b",
]

GRADER_MODEL = [
    "openai/gemma3:4b",
]

LANGUAGES = ["en", "de", "ar", "ch", "ta"]
```

### 6. Run all cells in order

*Runtime → Run all* (or run cells one by one). The cells will:

1. Mount Google Drive
2. Verify the GPU is available (`nvidia-smi`)
3. Write `config.py` from the configuration cell above
4. Install Ollama and start the service
5. Pull the configured models
6. Install Python dependencies (`inspect-ai`, `openai`, etc.)
7. Set `OPENAI_API_KEY=ollama` and `OPENAI_BASE_URL=http://localhost:11434/v1` to point at local Ollama
8. Run `eval.py` — results are written to `logs/` inside `PROJECT_PATH`

### 7. View results

Results are saved to `logs/` in your Drive folder. Download that folder and run locally:

```bash
inspect view
```

---

## Option B — Local run (using Open WebUI)

Use this if you have access to an Open WebUI instance with models already loaded.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up API credentials

Copy `.env.example` to `.env` and fill in your Open WebUI key and base URL:

```
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://your-open-webui-instance.example/api/v1
```

### 3. Configure models and languages

Edit `config.py` to enable the models and languages you want to test.

### 4. Run

```bash
python eval.py
```

### 5. View results

```bash
inspect view
```
