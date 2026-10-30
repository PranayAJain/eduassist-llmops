import mlflow, pandas as pd, config
from llm_utils import ask
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("eduassist_llmops_assignment")
NAME = "eduassist_prompt"

summary, details = [], []
for ver in [1, 2, 3]:
    p = mlflow.genai.load_prompt(f"prompts:/{NAME}/{ver}")
    label = f"v{ver}"
    rows = [ask(qid, q, label, p.template) for qid, q in config.QUESTIONS.items()]
    df = pd.DataFrame(rows); details += rows
    m = {"avg_latency_seconds": df.latency_seconds.mean(),
         "avg_input_tokens":    df.input_tokens.mean(),
         "avg_output_tokens":   df.output_tokens.mean(),
         "avg_cost_per_request":df.total_cost_usd.mean(),
         "valid_json":          df.valid_json.mean()}
    with mlflow.start_run(run_name=f"prompt_{label}"):
        mlflow.log_params({"prompt_name": NAME, "prompt_version": ver, "prompt_uri": f"prompts:/{NAME}/{ver}",
                           "model": config.MODEL, "api": config.API_STYLE, "temperature": 0,
                           "n_questions": len(df), "price_in_per_1M_usd": config.INPUT_PRICE_PER_M,
                           "price_out_per_1M_usd": config.OUTPUT_PRICE_PER_M, "pricing_date": config.PRICE_DATE})
        mlflow.log_metrics(m)
        mlflow.log_table(df, artifact_file=f"responses_{label}.json")
    summary.append({"version": label, **m})
    print(label, {k: round(v, 6) for k, v in m.items()})

pd.DataFrame(summary).to_csv("prompt_evaluation_results.csv", index=False)
pd.DataFrame(details).to_csv("prompt_evaluation_details.csv", index=False)
print("\nsaved prompt_evaluation_results.csv")
