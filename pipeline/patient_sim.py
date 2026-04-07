"""
patient_sim.py
Patient simulator — uses GPT-4o to roleplay as the patient,
responding only from the structured case features provided.
"""

import os
from models import call_gpt4o


SIMULATOR_SYSTEM_PROMPT = """You are roleplaying as a patient presenting to a primary care clinic.
You have the following clinical features. Answer the doctor's questions naturally,
in plain non-medical first-person language, as a real patient would.

Rules you must follow:
1. Only share information you have been ASKED about. Do not volunteer anything else.
2. Do not use medical terminology — speak as a lay person.
3. If asked about a symptom or feature not in your case description, say something like
   "I haven't noticed anything like that" or "Not that I'm aware of."
4. Never reveal your diagnosis, even if you know it.
5. Keep answers concise — 1 to 3 sentences.
6. If the doctor asks a yes/no question, answer yes or no first, then briefly elaborate.

Your clinical features:
{case_features}
"""


def build_case_features_text(case_row: dict) -> str:
    """Convert a CSV row into a readable feature list for the simulator."""
    feature_fields = [
        ("chief_complaint",        "Chief complaint"),
        ("duration_symptoms",      "Duration of symptoms"),
        ("location",               "Location"),
        ("size_description",       "Size description"),
        ("reducible",              "Reducible"),
        ("pain_character",         "Pain character"),
        ("pain_severity",          "Pain severity (0-10)"),
        ("aggravating_factors",    "Aggravating factors"),
        ("relieving_factors",      "Relieving factors"),
        ("nausea_vomiting",        "Nausea/vomiting"),
        ("bowel_changes",          "Bowel changes"),
        ("fever",                  "Fever"),
        ("prior_hernia_repair",    "Prior hernia repair"),
        ("prior_abdominal_surgery","Prior abdominal surgery"),
        ("anticoagulation",        "On anticoagulation"),
        ("pregnancy",              "Pregnancy"),
        ("bmi_category",           "BMI category"),
        ("relevant_comorbidities", "Relevant comorbidities"),
    ]
    lines = []
    for col, label in feature_fields:
        val = case_row.get(col, "")
        if val and str(val).strip() and str(val).strip().lower() not in ("nan", "none", ""):
            lines.append(f"- {label}: {val}")
    return "\n".join(lines) if lines else "No features provided."


class PatientSimulator:
    def __init__(self, case_row: dict):
        self.case_row = case_row
        self.case_features = build_case_features_text(case_row)
        self.system_prompt = SIMULATOR_SYSTEM_PROMPT.format(
            case_features=self.case_features
        )
        self.history: list[dict] = []
        self.total_input_tokens  = 0
        self.total_output_tokens = 0

    def respond(self, doctor_question: str) -> tuple[str, int, int]:
        """
        Given a doctor's question, return the patient's response.
        Returns (patient_response, input_tokens, output_tokens).
        """
        self.history.append({"role": "user", "content": doctor_question})

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.history,
        ]

        response, in_tok, out_tok = call_gpt4o(messages)
        self.history.append({"role": "assistant", "content": response})
        self.total_input_tokens  += in_tok
        self.total_output_tokens += out_tok
        return response, in_tok, out_tok

    def reset(self):
        """Reset conversation history for a new run."""
        self.history = []
        self.total_input_tokens  = 0
        self.total_output_tokens = 0
