import time, json
from datetime import datetime
import config

PROMPTS = {
 "v1": "Answer the following student question clearly.\nQuestion: {{question}}",
 "v2": ("You are an academic assistant.\nAnswer the following student question in a clear, "
        "concise, and accurate manner.\nUse simple language suitable for undergraduate students.\n"
        "Question: {{question}}"),
 "v3": ("You are an academic assistant.\nAnswer the student question and return ONLY valid JSON.\n"
        "Do not include markdown, explanations outside JSON, or code fences.\nUse this exact JSON schema:\n"
        "{\n  \"answer\": \"string\",\n  \"confidence\": \"high | medium | low\",\n  \"key_points\": [\"string\"]\n}\n"
        "Question: {{question}}"),
}

def cost(in_tok, out_tok):
    ic = in_tok / 1e6 * config.INPUT_PRICE_PER_M
    oc = out_tok / 1e6 * config.OUTPUT_PRICE_PER_M
    return ic, oc, ic + oc

def is_valid_json(text):
    try:
        d = json.loads(text)
        return isinstance(d, dict) and {"answer", "confidence", "key_points"} <= d.keys()
    except Exception:
        return False

def ask(qid, question, version, template):
    prompt = template.replace("{{question}}", question)
    row = {"question_id": qid, "prompt_version": version, "model": config.MODEL,
           "start_time": datetime.now().isoformat(timespec="milliseconds")}
    t0 = time.perf_counter()
    try:
        r = config.client.chat.completions.create(model=config.MODEL,
              messages=[{"role": "user", "content": prompt}], max_tokens=1024, temperature=0)
        text, in_tok, out_tok, status = (r.choices[0].message.content or "",
              r.usage.prompt_tokens, r.usage.completion_tokens, "success")
    except Exception as e:
        text, in_tok, out_tok, status = "", 0, 0, f"error: {type(e).__name__}"
    lat = time.perf_counter() - t0
    row["end_time"] = datetime.now().isoformat(timespec="milliseconds")
    ic, oc, tc = cost(in_tok, out_tok)
    row.update({"latency_seconds": round(lat, 3), "status": status,
                "input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": in_tok + out_tok,
                "input_cost_usd": ic, "output_cost_usd": oc, "total_cost_usd": tc,
                "valid_json": is_valid_json(text), "response": text})
    return row
