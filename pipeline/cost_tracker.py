"""
cost_tracker.py
Tracks token usage and API cost per interaction.
Pricing is per 1M tokens (input/output) as of mid-2025.
Update prices here if APIs change their rates.
"""

import csv
import os
from datetime import datetime
from dataclasses import dataclass, field

# ── Pricing table (USD per 1M tokens) ──────────────────────────────────────
PRICING = {
    "gpt-4o": {
        "input":  2.50,
        "output": 10.00,
    },
    "gemini-1.5-pro": {
        "input":  1.25,
        "output": 5.00,
    },
    "claude-sonnet-4-20250514": {
        "input":  3.00,
        "output": 15.00,
    },
    # Patient simulator (always GPT-4o)
    "patient-simulator": {
        "input":  2.50,
        "output": 10.00,
    },
}


@dataclass
class RunCost:
    case_id: str
    model: str
    prompt_type: str
    phase: int
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def calculate(self):
        key = self.model if self.model in PRICING else "gpt-4o"
        p = PRICING[key]
        self.cost_usd = (
            (self.input_tokens / 1_000_000) * p["input"] +
            (self.output_tokens / 1_000_000) * p["output"]
        )
        return self.cost_usd


class CostTracker:
    def __init__(self, output_path: str = "outputs/results/cost_log.csv"):
        self.output_path = output_path
        self.runs: list[RunCost] = []
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def log(self, case_id, model, prompt_type, phase,
            input_tokens, output_tokens) -> float:
        run = RunCost(
            case_id=case_id,
            model=model,
            prompt_type=prompt_type,
            phase=phase,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        run.calculate()
        self.runs.append(run)
        return run.cost_usd

    def save(self):
        """Write per-interaction cost log."""
        if not self.runs:
            return
        fieldnames = ["timestamp", "case_id", "model", "prompt_type",
                      "phase", "input_tokens", "output_tokens", "cost_usd"]
        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.runs:
                writer.writerow({
                    "timestamp":     r.timestamp,
                    "case_id":       r.case_id,
                    "model":         r.model,
                    "prompt_type":   r.prompt_type,
                    "phase":         r.phase,
                    "input_tokens":  r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "cost_usd":      round(r.cost_usd, 6),
                })

    def summary(self) -> dict:
        """Return cost totals grouped by model."""
        totals: dict[str, float] = {}
        for r in self.runs:
            totals[r.model] = totals.get(r.model, 0.0) + r.cost_usd
        grand = sum(totals.values())
        return {"by_model": totals, "grand_total_usd": round(grand, 4)}

    def save_summary(self, path: str = "outputs/results/cost_summary.csv"):
        summary = self.summary()
        rows = [{"model": m, "total_cost_usd": round(c, 4)}
                for m, c in summary["by_model"].items()]
        rows.append({"model": "TOTAL", "total_cost_usd": summary["grand_total_usd"]})
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "total_cost_usd"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n💰 Cost summary saved → {path}")
        print(f"   Grand total: ${summary['grand_total_usd']:.4f} USD")
