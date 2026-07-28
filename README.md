# Attacking LLMs with Low-Resource Languages

## Bachelor Thesis Project

**Students:** Caroline Lampart, Raphael Monteiro

**Supervisor:** Dr. Rebecca Balebako

---

## Research Question

> Do LLMs respond more safely to attacks in high resource languages compared to low resource languages?

## Description

This Bachelor thesis conducts red-teaming AI research to understand whether LLMs are easier to attack when prompts are written in different human languages.

Using [Inspect AI](https://inspect.ai-safety-institute.org.uk/) and Python, we automate single-turn adversarial prompt testing across five languages spanning from high-resource to low-resource: English, Arabic, German, Swiss German, and Tamil. The dataset consists of 50 cybersecurity-focused prompts drawn from the LinguaSafe and XSTest datasets.

Three target models are evaluated (DeepSeek, Gemma3, Apertus) using Mistral as an LLM-as-judge with yes/no labelling and confidence scoring. Experiments are run 15 times; results are reported as Attack Success Rate (ASR) and Over-Refusal Rate (ORR).

## Dataset

The pipeline is designed to run at multiple scales. Available dataset sizes: **1, 10, 50, 100, and 300 samples**. 

The primary evaluation uses 50 prompts (35 from LinguaSafe, 15 from XSTest).

Two sources are combined:

| Source | Type | Metric |
| --- | --- | --- |
| LinguaSafe | Malicious prompts | Attack Success Rate (ASR) |
| XSTest | Safe prompts | Over-Refusal Rate (ORR) |

ASR measures how often a model complies with a harmful request; ORR measures how often a model unnecessarily refuses a benign request.

## Repository Structure

| Path | Contents |
| --- | --- |
| [`src/`](src) | Experiment pipeline and evaluation scripts |

## Reproducing the Experiments

See [`src/README.md`](src) for setup instructions (local run against an
OpenAI-compatible API, or Google Colab with Ollama).

## Data & Licensing

Prompts in `src/data/` are drawn from the LinguaSafe and XSTest datasets and
remain subject to their original licenses and attribution terms. All other
code, data, and documentation in this repository is licensed under
[CC BY-NC-SA 4.0](LICENSE) (Attribution-NonCommercial-ShareAlike) — free to
share and adapt for non-commercial purposes, with attribution, under the
same license.
