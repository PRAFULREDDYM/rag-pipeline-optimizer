# backend/evaluator.py
from typing import List, Dict, Any
from groq import Groq
import os
import json
from dotenv import load_dotenv
load_dotenv()
GROQ_MODEL = "llama-3.1-8b-instant"


class GroqEvaluator:
    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        self.client = Groq(api_key=api_key)

    def evaluate(
        self, question: str, pipeline_answers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Ask Groq to act as a judge.
        Returns list with scores merged into each pipeline dict.
        """

        desc_lines = [f"User question: {question}", ""]
        for p in pipeline_answers:
            desc_lines.append(f"Pipeline {p['pipeline_id']} ({p['pipeline_name']}):")
            desc_lines.append(f"Answer: {p['answer']}")
            desc_lines.append("")
        desc_text = "\n".join(desc_lines)

        judge_prompt = f"""
You are an evaluation agent.

Given a question and several pipeline answers, score each pipeline on:
- accuracy (0-10)
- relevance (0-10)

Return STRICT JSON with this shape:

{{
  "pipelines": [
    {{"id": "A", "accuracy": 0-10, "relevance": 0-10}},
    ...
  ]
}}

Question and answers:
{desc_text}
"""

        completion = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict JSON-generating evaluation agent.",
                },
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0,
        )
        raw = completion.choices[0].message.content

        try:
            data = json.loads(raw)
            scores_by_id = {p["id"]: p for p in data.get("pipelines", [])}
        except Exception:
            # Fallback: neutral scores if parsing fails
            scores_by_id = {
                p["pipeline_id"]: {"accuracy": 5, "relevance": 5}
                for p in pipeline_answers
            }

        results = []
        for p in pipeline_answers:
            pid = p["pipeline_id"]
            score = scores_by_id.get(pid, {"accuracy": 5, "relevance": 5})

            tokens = p.get("approx_tokens", 0)
            cost_score = max(0.0, 10.0 - tokens / 500.0)

            overall = (
                0.45 * score["accuracy"]
                + 0.45 * score["relevance"]
                + 0.10 * cost_score
            )

            p["scores"] = {
                "accuracy": score["accuracy"],
                "relevance": score["relevance"],
                "cost": round(cost_score, 2),
                "overall": round(overall, 2),
            }
            results.append(p)

        return results