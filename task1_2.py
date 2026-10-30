import json
import pandas as pd
import config
from llm_utils import ask, PROMPTS

rows = [ask(qid, q, "v3", PROMPTS["v3"]) for qid, q in config.QUESTIONS.items()]
df = pd.DataFrame(rows)
df.drop(columns=["response"]).round(8).to_csv("latency_tokens_cost.csv", index=False)
json.dump(rows, open("task1_responses.json", "w"), indent=2)

pd.set_option("display.width", 200)
print(df[["question_id", "prompt_version", "model", "start_time", "end_time", "latency_seconds", "status"]].to_string(index=False))
s = df["latency_seconds"]
print(f"\nmean={s.mean():.3f} median={s.median():.3f} min={s.min():.3f} max={s.max():.3f} std={s.std():.3f}")
print("slowest:", df.loc[s.idxmax(), "question_id"])

print()
print(df[["question_id", "input_tokens", "output_tokens", "total_tokens", "total_cost_usd"]].to_string(index=False))
avg = df["total_cost_usd"].mean()
print(f"\navg cost per request: ${avg:.8f}")
print(f"monthly ({config.MONTHLY_REQUESTS} requests): ${avg * config.MONTHLY_REQUESTS:.4f}")
print("valid json:", df["valid_json"].sum(), "/", len(df))
