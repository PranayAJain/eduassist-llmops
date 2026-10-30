import nbformat as nbf
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# EduAssist AI - LLMOps: API Cost, Latency and Prompt Version Management
**Course:** AIMLCZG523 MLOps - Assignment 2  |  **Student ID:** 2025AE05025

| Item | Value |
|---|---|
| LLM | `gemini-3.5-flash-lite` (Google Gemini API) |
| API | Gemini API `v1beta`, OpenAI-compatible endpoint, called with `openai` Python SDK 3.41.0 |
| Pricing | USD 0.30 / 1M input tokens, USD 2.50 / 1M output tokens (Standard paid tier) |
| Pricing source / date | ai.google.dev/gemini-api/docs/pricing, checked 30-Oct-2026, currency USD |
| Tracking | MLflow 3.3.2 (SQLite backend), experiment `eduassist_llmops_assignment`, prompt `eduassist_prompt` |

The API key is read from the `GEMINI_API_KEY` environment variable. Costs are computed at the paid-tier rates above.
""")

code("""
import os, json, difflib
import pandas as pd, mlflow
import config
from llm_utils import ask, PROMPTS

RERUN = False
assert "GEMINI_API_KEY" in os.environ
mlflow.set_tracking_uri("http://127.0.0.1:5000")
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 80)
print("model:", config.MODEL, "| endpoint:", config.BASE_URL)
print(f"pricing ({config.CURRENCY}, {config.PRICE_DATE}): in ${config.INPUT_PRICE_PER_M}/1M, out ${config.OUTPUT_PRICE_PER_M}/1M")
""")

md("### llm_utils.py")
code("""!cat llm_utils.py""")

# ---------------- TASK 1 ----------------
md("""
## Task 1 - API Latency Measurement
5 questions, prompt v3, `temperature=0`. Latency measured with `time.perf_counter()` around each API call.
""")
code("""
if RERUN or not os.path.exists("latency_tokens_cost.csv"):
    rows = [ask(qid, q, "v3", PROMPTS["v3"]) for qid, q in config.QUESTIONS.items()]
    pd.DataFrame(rows).drop(columns=["response"]).to_csv("latency_tokens_cost.csv", index=False)
    json.dump(rows, open("task1_responses.json", "w"), indent=2)
df1 = pd.read_csv("latency_tokens_cost.csv")
df1[["question_id","prompt_version","model","start_time","end_time","latency_seconds","status"]]
""")
code("""
s = df1["latency_seconds"]
stats = pd.DataFrame({"statistic": ["mean","median","min","max","std"],
                      "latency_seconds": [s.mean(), s.median(), s.min(), s.max(), s.std()]}).round(3)
print("slowest request:", df1.loc[s.idxmax(), "question_id"], "| fastest:", df1.loc[s.idxmin(), "question_id"])
stats
""")
code("""
for r in json.load(open("task1_responses.json")):
    d = json.loads(r["response"])
    print(f"{r['question_id']}: {config.QUESTIONS[r['question_id']]}\\n   confidence={d['confidence']} | {d['answer'][:150]}...\\n")
""")
md("""
**Interpretation.** All 5 requests succeeded. Mean latency was 1.214 s (median 1.213 s) with a small spread
(std 0.132 s, range 1.007-1.333 s). The slowest request was **Q5** ("Explain gradient descent in simple terms", 1.333 s).
It also produced the most output tokens (171). An LLM generates its answer one token at a time, so longer, more
explanatory answers take longer, and "explain in simple terms" invites a longer answer. Token count is not the
whole story, though. Q3 generated only 115 tokens yet took 1.325 s, while Q1 generated 168 tokens in 1.213 s. With
answers this short, network round-trip and server-side queueing on a shared, free-tier endpoint add jitter of about
0.1-0.3 s, which is as large as the effect of answer length. With only 5 samples these figures are indicative; production
monitoring should track p95/p99 latency over many requests.
""")

# ---------------- TASK 2 ----------------
md("""
## Task 2 - Token Usage and Cost Estimation
Input Cost = input_tokens / 1,000,000 x 0.30 USD; Output Cost = output_tokens / 1,000,000 x 2.50 USD;
Total = Input + Output; Projected monthly cost = average cost per request x 10,000 requests.
""")
code("""
t2 = df1[["question_id","input_tokens","output_tokens","total_tokens","input_cost_usd","output_cost_usd","total_cost_usd"]]
t2.style.format({c: "{:.6f}" for c in ["input_cost_usd","output_cost_usd","total_cost_usd"]})
""")
code("""
avg = df1["total_cost_usd"].mean()
out_share = df1["output_cost_usd"].sum() / df1["total_cost_usd"].sum()
print(f"Average cost per request : ${avg:.8f}")
print(f"Projected monthly cost   : ${avg*config.MONTHLY_REQUESTS:.4f}  ({config.MONTHLY_REQUESTS:,} requests/month, USD)")
print(f"Share of cost from output tokens: {out_share:.1%}")
""")
md("""
**Cost-optimisation recommendation.** Output tokens cause about 93% of the cost: they are 8.3x more expensive than
input tokens, and the model writes more than it reads. The most effective lever is therefore to **limit the answer
length**. Add a length rule to the production prompt, such as "answer in at most 80 words and give at most 3
key_points", and set `max_tokens` to around 250 as a safety cap. Cutting the average output from about 139 to about 70
tokens would lower cost per request by about 46%, and it would also reduce latency, because generation time grows with
output length. Repeated questions (for example common course FAQs) could additionally be served from a response cache,
which avoids an API call altogether.
""")

# ---------------- TASK 3 ----------------
md("""
## Task 3 - Prompt Versioning and Evaluation with MLflow
### 3.1 Register prompts v1, v2, v3 in the MLflow Prompt Registry
""")
code("""
NAME = "eduassist_prompt"
MSG = {"v1": "v1: baseline - minimal instruction, free-text answer",
       "v2": "v2: add academic-assistant role, clarity and undergraduate-level language",
       "v3": "v3: enforce strict JSON output schema (answer, confidence, key_points), no markdown"}
try:
    mlflow.genai.load_prompt(f"prompts:/{NAME}/3"); print("prompt versions already registered - skipping")
except Exception:
    for v in ["v1", "v2", "v3"]:
        p = mlflow.genai.register_prompt(name=NAME, template=PROMPTS[v], commit_message=MSG[v],
                                         tags={"label": v, "app": "EduAssist"})
        print("registered", NAME, "version", p.version)
for ver in [1, 2, 3]:
    p = mlflow.genai.load_prompt(f"prompts:/{NAME}/{ver}")
    print(f"\\n--- version {p.version} | commit: {p.commit_message}\\n{p.template}")
""")
md("### 3.2 Prompt diff")
code("""
tpl = {v: mlflow.genai.load_prompt(f"prompts:/{NAME}/{v}").template for v in [1, 2, 3]}
for a, b in [(1, 2), (2, 3)]:
    print(f"===== diff v{a} -> v{b} =====")
    print("\\n".join(difflib.unified_diff(tpl[a].splitlines(), tpl[b].splitlines(), f"v{a}", f"v{b}", lineterm="")), "\\n")
""")
md("""
### 3.3 Evaluation runs
`valid_json` = responses that parse as JSON with keys answer, confidence, key_points / total responses.
""")
code("""
mlflow.set_experiment("eduassist_llmops_assignment")
existing = mlflow.search_runs(experiment_names=["eduassist_llmops_assignment"])
if RERUN or len(existing) < 3:
    summary, details = [], []
    for ver in [1, 2, 3]:
        p = mlflow.genai.load_prompt(f"prompts:/{NAME}/{ver}"); label = f"v{ver}"
        df = pd.DataFrame([ask(qid, q, label, p.template) for qid, q in config.QUESTIONS.items()])
        details += df.to_dict("records")
        m = {"avg_latency_seconds": df.latency_seconds.mean(), "avg_input_tokens": df.input_tokens.mean(),
             "avg_output_tokens": df.output_tokens.mean(), "avg_cost_per_request": df.total_cost_usd.mean(),
             "valid_json": df.valid_json.mean()}
        with mlflow.start_run(run_name=f"prompt_{label}"):
            mlflow.log_params({"prompt_name": NAME, "prompt_version": ver, "prompt_uri": f"prompts:/{NAME}/{ver}",
                               "model": config.MODEL, "api": config.API_STYLE, "temperature": 0,
                               "n_questions": len(df), "price_in_per_1M_usd": config.INPUT_PRICE_PER_M,
                               "price_out_per_1M_usd": config.OUTPUT_PRICE_PER_M, "pricing_date": config.PRICE_DATE})
            mlflow.log_metrics(m)
            mlflow.log_table(df, artifact_file=f"responses_{label}.json")
        summary.append({"version": label, **m})
    pd.DataFrame(summary).to_csv("prompt_evaluation_results.csv", index=False)
    pd.DataFrame(details).to_csv("prompt_evaluation_details.csv", index=False)
else:
    print("using the 3 runs already logged in MLflow")

runs = mlflow.search_runs(experiment_names=["eduassist_llmops_assignment"], order_by=["tags.mlflow.runName"])
runs[["tags.mlflow.runName","run_id","metrics.avg_latency_seconds","metrics.avg_input_tokens",
      "metrics.avg_output_tokens","metrics.avg_cost_per_request","metrics.valid_json"]]
""")
md("### 3.4 Comparison table")
code("""
res = pd.read_csv("prompt_evaluation_results.csv")
best = res.sort_values(["valid_json", "avg_cost_per_request"], ascending=[False, True]).iloc[0]["version"]
cmp = pd.DataFrame({"Version": res.version,
    "Avg Latency (s)": res.avg_latency_seconds.round(3),
    "Avg Input Tokens": res.avg_input_tokens.round(1),
    "Avg Output Tokens": res.avg_output_tokens.round(1),
    "Avg Cost (USD)": res.avg_cost_per_request.map("{:.6f}".format),
    "Valid JSON Rate": res.valid_json.map("{:.0%}".format),
    "Recommended?": ["Yes" if v == best else "No" for v in res.version]})
cmp
""")
code("""
det = pd.read_csv("prompt_evaluation_details.csv")
det.pivot(index="question_id", columns="prompt_version", values="output_tokens").add_prefix("out_tokens_")
""")
md("### 3.5 Promote the best version with the alias `production`")
code("""
mlflow.genai.set_prompt_alias(NAME, alias="production", version=int(best[1:]))
p = mlflow.genai.load_prompt(f"prompts:/{NAME}@production")
print(f"prompts:/{NAME}@production -> version {p.version} ({p.commit_message})")
""")
md("""
### Promotion justification
Version 3 was promoted to `production` because it is the only version that meets the EduAssist product requirement
and it is also the most efficient. The application expects a machine-readable JSON object with `answer`, `confidence`
and `key_points`. v3 produced valid JSON for 5 of 5 questions (100%), while v1 and v2 returned free-form text
and scored 0%, so their output would break the downstream parser no matter how good the content is.

v3 is also the best on cost and latency. Its prompt is the longest (87.2 input tokens versus 22.2 for v1), but the
fixed schema keeps the model's answers short: 149.4 output tokens on average versus 461.8 for v1 and 308.4 for v2.
Output tokens are priced 8.3x higher than input tokens, so v3 costs USD 0.000400 per request, which is 66% cheaper
than v1 (USD 0.001161) and 49% cheaper than v2 (USD 0.000785). At 10,000 requests a month that is about USD 4.00
versus USD 11.61 for v1. Shorter generation also halves latency, from 2.18 s for v1 to 1.10 s.

The trade-off is that v3's answers are more condensed than v2's explanatory text, and `confidence` is self-reported by
the model. Using a registry alias means the application loads `prompts:/eduassist_prompt@production`, so a future
v4 can be promoted, or v3 rolled back, by moving the alias without any code change.
""")

md("### MLflow UI screenshots")
code("""
from IPython.display import Image, display, Markdown
for f in sorted(os.listdir("screenshots")):
    if f.endswith(".png"):
        display(Markdown(f"**{f}**")); display(Image(filename=os.path.join("screenshots", f), width=900))
""")
md("""
## Conclusion
`gemini-3.5-flash-lite` answered the 5 academic questions in about 1.2 s each, at about USD 0.00037 per request
(about USD 3.74 per month for 10,000 requests). Prompt versioning in MLflow showed that the prompt with the strict JSON
schema (v3) is the most reliable, cheapest and fastest, and it now serves as the `production` alias.
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "EduAssist_LLMOps_Assignment2.ipynb")
print("notebook written:", len(C), "cells")
