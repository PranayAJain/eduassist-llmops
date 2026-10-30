import mlflow
from llm_utils import PROMPTS
mlflow.set_tracking_uri("http://127.0.0.1:5000")
NAME = "eduassist_prompt"
MSG = {
 "v1": "v1: baseline - minimal instruction, free-text answer",
 "v2": "v2: add academic-assistant role, clarity and undergraduate-level language",
 "v3": "v3: enforce strict JSON output schema (answer, confidence, key_points), no markdown",
}
for v in ["v1", "v2", "v3"]:
    p = mlflow.genai.register_prompt(name=NAME, template=PROMPTS[v], commit_message=MSG[v],
                                     tags={"label": v, "app": "EduAssist"})
    print(f"registered {NAME} version {p.version}  <- {v}")
