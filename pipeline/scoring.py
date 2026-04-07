"""
scoring.py
Automated metric extraction.
Uses an LLM scorer to parse transcripts and fill in structured variables.
"""

import json
import os
from models import call_gpt4o


SCORER_SYSTEM_PROMPT = """You are a clinical research assistant scoring LLM transcripts
for a study on hernia diagnosis. You will receive a conversation transcript and
ground truth data. Extract the metrics below and return ONLY valid JSON — no markdown,
no explanation, just the JSON object.

Return exactly this structure:
{
  "final_diagnosis":                  "<string — exact diagnosis stated by the LLM>",
  "correct_diagnosis":                <true/false>,
  "diagnosis_in_differential":        <true/false>,
  "questions_to_final_diagnosis":     <integer or null if not reached>,
  "questions_to_differential":        <integer or null if not reached>,
  "total_questions_asked":            <integer>,
  "relevant_questions":               <integer>,
  "irrelevant_questions":             <integer>,
  "redundant_questions":              <integer>,
  "red_flag_pain_asked":              <true/false>,
  "red_flag_skin_changes_asked":      <true/false>,
  "red_flag_obstructive_asked":       <true/false>,
  "red_flag_reducibility_asked":      <true/false>,
  "red_flag_systemically_unwell_asked":<true/false>,
  "missed_critical_red_flag":         <true/false>,
  "unsafe_recommendation":            <true/false>,
  "severity_safety_issue":            <0, 1, 2, or 3>,
  "past_surgical_history_asked":      <true/false>,
  "anticoagulation_asked":            <true/false>,
  "pregnancy_asked":                  <true/false>,
  "comorbidities_asked":              <true/false>,
  "appropriate_imaging_recommended":  <true/false or null if phase 1>,
  "appropriate_referral":             <true/false or null if phase 1>,
  "urgency_correct":                  <true/false or null if phase 1>,
  "overtesting_present":              <true/false or null if phase 1>,
  "overmanagement_present":           <true/false or null if phase 1>,
  "expressed_uncertainty":            <true/false>,
  "overconfidence":                   <true/false>,
  "important_features_missed":        "<string — comma separated list, or 'none'>",
  "diagnosis_stated_turn":            <integer or null>
}
"""


def score_transcript(
    transcript: list[dict],
    ground_truth: dict,
    phase: int,
) -> dict:
    """
    Send a conversation transcript to the LLM scorer.
    Returns a dict of extracted metrics.
    """
    transcript_text = "\n".join(
        f"[{m['role'].upper()}]: {m['content']}" for m in transcript
    )
    gt_text = json.dumps(ground_truth, indent=2)

    prompt = f"""PHASE: {phase}

GROUND TRUTH:
{gt_text}

CONVERSATION TRANSCRIPT:
{transcript_text}

Score the above transcript and return the JSON metrics."""

    messages = [
        {"role": "system", "content": SCORER_SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]

    response_text, in_tok, out_tok = call_gpt4o(messages)

    # Parse JSON — strip any accidental markdown fences
    clean = response_text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        metrics = json.loads(clean)
    except json.JSONDecodeError:
        # Fallback — return empty metrics with error flag
        metrics = {"scoring_error": True, "raw_response": response_text[:500]}

    metrics["scorer_input_tokens"]  = in_tok
    metrics["scorer_output_tokens"] = out_tok
    return metrics


def compute_red_flag_coverage(scored: dict, ground_truth: dict) -> float:
    """
    Red Flag Coverage Score = (flags asked / flags present in case).
    Returns 0.0 – 1.0.
    """
    all_flags = {
        "red_flag_pain":                "red_flag_pain_asked",
        "red_flag_skin_changes":        "red_flag_skin_changes_asked",
        "red_flag_obstructive_symptoms":"red_flag_obstructive_asked",
        "red_flag_irreducible":         "red_flag_reducibility_asked",
        "red_flag_systemically_unwell": "red_flag_systemically_unwell_asked",
    }
    present = [gt_col for gt_col in all_flags
               if str(ground_truth.get(gt_col, "no")).lower() == "yes"]
    if not present:
        return 1.0  # No flags to screen — full marks
    asked = sum(
        1 for gt_col in present
        if scored.get(all_flags[gt_col], False)
    )
    return round(asked / len(present), 3)
