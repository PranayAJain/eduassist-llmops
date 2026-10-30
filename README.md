# EduAssist AI - LLMOps Assignment 2 (2025AE05025)

API latency, token cost and prompt version management for an academic Q&A assistant using Gemini and MLflow.

## Environment
- Python 3.11 (conda env `llmops`)
- Libraries: see `requirements.txt` (mlflow 3.3.2, openai, pandas, sqlalchemy 2.0.36, ipykernel, nbconvert)
- Model: `gemini-3.5-flash-lite`, Gemini API v1beta (OpenAI-compatible endpoint)
- Pricing used: USD 0.30 / 1M input tokens, USD 2.50 / 1M output tokens (ai.google.dev pricing page, 30-Oct-2026)

## Environment variables
| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio API key (never stored in code or files) |

## Install
```bash
conda create -y -n llmops python=3.11
conda activate llmops
pip install -r requirements.txt
export GEMINI_API_KEY=<your key>
```

## MLflow UI
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```
Open http://127.0.0.1:5000 - experiment `eduassist_llmops_assignment`, prompt `eduassist_prompt` (alias `production` -> version 3).

## Reproduce
1. Start the MLflow server (above).
2. `python task1_2.py` - Task 1 and 2, writes `latency_tokens_cost.csv`.
3. `python task3_register.py` - registers prompt versions v1-v3.
4. `python task3_evaluate.py` - evaluates v1-v3, logs runs, writes `prompt_evaluation_results.csv`.
5. Promote: `mlflow.genai.set_prompt_alias("eduassist_prompt", alias="production", version=3)`.
6. Or run everything from the notebook `EduAssist_LLMOps_Assignment2.ipynb` with `RERUN = True`.

## Files
| File | Content |
|---|---|
| `EduAssist_LLMOps_Assignment2.ipynb` / `.html` | Notebook with code, outputs and analysis |
| `config.py`, `llm_utils.py` | Model/pricing settings and API call helper |
| `task1_2.py`, `task3_register.py`, `task3_evaluate.py` | Standalone task scripts |
| `latency_tokens_cost.csv` | Task 1 and 2 results per question |
| `prompt_evaluation_results.csv` | Task 3 metrics per prompt version |
| `prompt_evaluation_details.csv` | Task 3 per-question results |
| `screenshots/` | MLflow experiment, metrics, prompt registry, diff, production alias |
| `Report_2025AE05025.pdf` | Final report |
