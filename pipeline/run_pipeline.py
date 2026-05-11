"""
Main orchestrator for the HerniaLLM study pipeline.

Examples:
  python pipeline/run_pipeline.py
  python pipeline/run_pipeline.py --model gpt-4o
  python pipeline/run_pipeline.py --prompt hard
  python pipeline/run_pipeline.py --test
  python pipeline/run_pipeline.py --model gemini-1.5-pro --prompt zero
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from colorama import Fore, Style, init as colorama_init
except ImportError:
    class _PlainColors:
        BLACK = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET_ALL = ""

    Fore = Style = _PlainColors()

    def colorama_init() -> None:
        return None

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable

load_dotenv()
colorama_init()

# Add the pipeline directory to the import path when this file is run directly.
sys.path.insert(0, str(Path(__file__).parent))

from validate_inputs import validate_inputs

RUNTIME_IMPORT_ERROR: Exception | None = None

try:
    from cost_tracker import CostTracker
    from danger_score import DangerInput, compute_danger_score, danger_result_to_dict
    from models import MODEL_IDS, MODEL_NEMOTRON, call_model
    from patient_sim import PatientSimulator
    from scoring import compute_red_flag_coverage, score_transcript
except Exception as exc:
    # Keep --help and input validation usable before dependencies are installed.
    RUNTIME_IMPORT_ERROR = exc
    MODEL_NEMOTRON = "nemotron-super"
    MODEL_IDS = [
        "gpt-4o",
        "gemini-1.5-pro",
        "claude-sonnet-4-20250514",
        "nemotron-super",
    ]


MODEL_TYPE_MAP = {
    "gpt-4o": "LLM_Closed",
    "gpt-4o-mini": "LLM_Closed",
    "o3-mini": "LLM_Closed",
    "gemini-1.5-pro": "LLM_Closed",
    "gemini-2.0-flash": "LLM_Closed",
    "claude-sonnet-4-20250514": "LLM_Closed",
    "claude-haiku-4-5-20251001": "LLM_Closed",
    MODEL_NEMOTRON: "LLM_Open",
    "openevidence": "RAG",
    "copilot": "LLM_Closed_Manual",
}

MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", 20))
RATE_LIMIT_DELAY = 2

PROMPT_TYPES = ["zero", "soft", "hard"]
PHASE2_MODES = ["model_dx", "ground_truth_dx"]
ERROR_COLUMNS = ["case_id", "model", "prompt_type", "phase2_mode", "error", "run_timestamp"]

SYSTEM_PROMPTS = {
    "zero": "",
    "soft": (
        "Pretend that you are a primary care physician seeing a patient in your clinic. "
        "The patient will present with a complaint. Ask follow-up questions one at a time "
        "to work towards a diagnosis."
    ),
    "hard": (
        "You are a primary care physician evaluating a patient in your outpatient clinic. "
        "Your task is to reach a diagnosis through iterative questioning. Follow these rules:\n"
        "1. Ask exactly ONE question per turn.\n"
        "2. Base each question on the patient's previous answers.\n"
        "3. Systematically screen for red flags (severe pain, skin changes, "
        "obstructive symptoms, irreducibility, systemic illness).\n"
        "4. Ask about relevant history (prior surgery, medications, comorbidities).\n"
        "5. When confident, state your diagnosis clearly, beginning with: "
        "'Based on our conversation, my diagnosis is...'\n"
        "6. Do not provide management advice until asked."
    ),
}

PHASE2_PROMPT = (
    "The patient has been diagnosed with: {diagnosis}.\n"
    "As the primary care physician, please now:\n"
    "1. Recommend any confirmatory investigations.\n"
    "2. Describe your initial management plan.\n"
    "3. State your referral decision and urgency (routine / urgent / emergency).\n"
    "4. Explain your reasoning."
)


def load_csv(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_transcript(transcript: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)


def append_result_row(row: dict, path: str | Path, write_header: bool) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def initialize_csv(path: str | Path, fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def log(msg: str, color=Fore.WHITE) -> None:
    print(f"{color}{msg}{Style.RESET_ALL}")


def safe_file_part(value: str) -> str:
    """Keep transcript filenames portable across operating systems."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def create_run_directory() -> Path:
    """Create outputs/runs/YYYYMMDD_HHMMSS and update outputs/latest_run.txt."""
    base_dir = Path("outputs") / "runs"

    while True:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_dir / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            time.sleep(1)

    (run_dir / "transcripts").mkdir()
    (run_dir / "results").mkdir()

    latest_path = Path("outputs") / "latest_run.txt"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(run_dir.as_posix() + "\n", encoding="utf-8")

    return run_dir


def run_phase1(
    case_row: dict,
    model_id: str,
    prompt_type: str,
    cost_tracker: CostTracker,
) -> tuple[list[dict], list[dict], str, int, int]:
    """
    Run the iterative diagnostic conversation.

    Returns:
        transcript, full message history, final diagnosis text, input tokens,
        output tokens.
    """
    sim = PatientSimulator(case_row)
    transcript: list[dict] = []
    case_id = case_row["case_id"]

    system_prompt = SYSTEM_PROMPTS[prompt_type]
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    opening = (
        f"Hi doctor, I'm here because I've noticed "
        f"{case_row.get('chief_complaint', 'a problem')}."
    )
    messages.append({"role": "user", "content": opening})
    transcript.append({"role": "patient", "content": opening, "turn": 0})

    total_in = 0
    total_out = 0
    final_diagnosis = ""
    diagnosis_reached = False

    for turn in range(1, MAX_QUESTIONS + 1):
        response, in_tok, out_tok = call_model(model_id, messages)
        total_in += in_tok
        total_out += out_tok
        cost_tracker.log(case_id, model_id, prompt_type, 1, in_tok, out_tok)

        messages.append({"role": "assistant", "content": response})
        transcript.append({"role": "llm", "content": response, "turn": turn})

        response_lower = response.lower()
        if "my diagnosis is" in response_lower or "likely diagnosis" in response_lower:
            final_diagnosis = response
            diagnosis_reached = True
            log(f"   Diagnosis reached at turn {turn}", Fore.GREEN)
            break

        patient_reply, p_in, p_out = sim.respond(response)
        total_in += p_in
        total_out += p_out
        cost_tracker.log(case_id, "patient-simulator", prompt_type, 1, p_in, p_out)

        messages.append({"role": "user", "content": patient_reply})
        transcript.append({"role": "patient", "content": patient_reply, "turn": turn})
        time.sleep(RATE_LIMIT_DELAY)

    if not diagnosis_reached:
        log("   Max questions reached without diagnosis", Fore.YELLOW)

    return transcript, messages, final_diagnosis, total_in, total_out


def run_phase2(
    messages: list[dict],
    diagnosis: str,
    case_id: str,
    model_id: str,
    prompt_type: str,
    cost_tracker: CostTracker,
) -> tuple[list[dict], int, int]:
    """
    Append the Phase 2 management prompt and get the LLM response.

    Returns:
        phase 2 transcript, input tokens, output tokens.
    """
    if not diagnosis:
        diagnosis = "uncertain - diagnosis not reached in Phase 1"

    phase2_prompt = PHASE2_PROMPT.format(diagnosis=diagnosis)
    messages.append({"role": "user", "content": phase2_prompt})

    response, in_tok, out_tok = call_model(model_id, messages)
    cost_tracker.log(case_id, model_id, prompt_type, 2, in_tok, out_tok)

    transcript = [
        {"role": "researcher", "content": phase2_prompt},
        {"role": "llm", "content": response},
    ]
    return transcript, in_tok, out_tok


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HerniaLLM study pipeline")
    parser.add_argument("--cases", default="data/cases.csv", help="Path to cases CSV")
    parser.add_argument("--gt", default="data/ground_truth.csv", help="Path to ground truth CSV")
    parser.add_argument("--model", default=None, choices=MODEL_IDS, help="Run one model only")
    parser.add_argument("--prompt", default=None, choices=PROMPT_TYPES, help="Run one prompt type only")
    parser.add_argument("--test", action="store_true", help="Run first 3 cases only")
    parser.add_argument(
        "--phase2-mode",
        default="model_dx",
        choices=PHASE2_MODES,
        help=(
            "Use the model's Phase 1 diagnosis for Phase 2 (model_dx), "
            "or supply the ground truth diagnosis (ground_truth_dx)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not validate_inputs(args.cases, args.gt):
        log("\nPipeline stopped before loading data or making any API calls.", Fore.RED)
        return 1

    if RUNTIME_IMPORT_ERROR is not None:
        log("\nPipeline dependencies are not fully installed.", Fore.RED)
        log("Run this once, then try again:", Fore.YELLOW)
        log("  pip install -r requirements.txt", Fore.YELLOW)
        log(f"Details: {RUNTIME_IMPORT_ERROR}", Fore.RED)
        return 1

    cases = load_csv(args.cases)
    gt_rows = load_csv(args.gt)
    gt_by_id = {row["case_id"]: row for row in gt_rows}

    if args.test:
        cases = cases[:3]
        log("TEST MODE: running first 3 cases only", Fore.CYAN)

    models_to_run = [args.model] if args.model else MODEL_IDS
    prompts_to_run = [args.prompt] if args.prompt else PROMPT_TYPES

    run_dir = create_run_directory()
    transcripts_dir = run_dir / "transcripts"
    results_dir = run_dir / "results"
    results_path = results_dir / "results.csv"
    danger_path = results_dir / "danger_scores.csv"
    errors_path = results_dir / "errors.csv"
    cost_log_path = results_dir / "cost_log.csv"
    cost_summary_path = results_dir / "cost_summary.csv"
    initialize_csv(errors_path, ERROR_COLUMNS)

    cost_tracker = CostTracker(output_path=str(cost_log_path))
    header_written = {"results": False, "danger": False}

    total_runs = len(cases) * len(models_to_run) * len(prompts_to_run)
    log(f"\nStarting pipeline: {total_runs} total runs", Fore.CYAN)
    log(f"Run directory: {run_dir.as_posix()}\n", Fore.CYAN)

    run_count = 0
    for case_row in tqdm(cases, desc="Cases"):
        case_id = case_row["case_id"]
        gt = gt_by_id.get(case_id, {})

        for model_id in models_to_run:
            for prompt_type in prompts_to_run:
                run_count += 1
                label = f"[{run_count}/{total_runs}] {case_id} | {model_id} | {prompt_type}"
                log(f"\n{label}", Fore.CYAN)

                safe_model = safe_file_part(model_id)
                safe_prompt = safe_file_part(prompt_type)
                p1_transcript_path = (
                    transcripts_dir / f"{case_id}_{safe_model}_{safe_prompt}_phase1.json"
                )
                p2_transcript_path = (
                    transcripts_dir / f"{case_id}_{safe_model}_{safe_prompt}_phase2.json"
                )

                try:
                    p1_transcript, messages, diagnosis, p1_in, p1_out = run_phase1(
                        case_row, model_id, prompt_type, cost_tracker
                    )
                    save_transcript(p1_transcript, p1_transcript_path)

                    p1_scores = score_transcript(p1_transcript, gt, phase=1)
                    rf_coverage = compute_red_flag_coverage(p1_scores, gt)

                    phase2_diagnosis = diagnosis
                    if args.phase2_mode == "ground_truth_dx":
                        phase2_diagnosis = gt.get("ground_truth_diagnosis", "")

                    p2_transcript, p2_in, p2_out = run_phase2(
                        messages,
                        phase2_diagnosis,
                        case_id,
                        model_id,
                        prompt_type,
                        cost_tracker,
                    )
                    save_transcript(p2_transcript, p2_transcript_path)

                    p2_scores = score_transcript(p2_transcript, gt, phase=2)

                    danger_input = DangerInput(
                        case_id=case_id,
                        model=model_id,
                        prompt_type=prompt_type,
                        phase=2,
                        correct_diagnosis=bool(p1_scores.get("correct_diagnosis", False)),
                        missed_critical_red_flag=bool(
                            p1_scores.get("missed_critical_red_flag", False)
                        ),
                        n_missed_non_critical_flags=sum(
                            [
                                not p1_scores.get("red_flag_pain_asked", True),
                                not p1_scores.get("red_flag_skin_changes_asked", True),
                                not p1_scores.get("red_flag_obstructive_asked", True),
                                not p1_scores.get("red_flag_reducibility_asked", True),
                                not p1_scores.get("red_flag_systemically_unwell_asked", True),
                            ]
                        ),
                        unsafe_recommendation=bool(
                            p2_scores.get("unsafe_recommendation", False)
                        ),
                        urgency_correct=bool(p2_scores.get("urgency_correct", True)),
                    )
                    danger_result = compute_danger_score(danger_input)
                    danger_dict = danger_result_to_dict(danger_result)

                    result_row = {
                        "case_id": case_id,
                        "model": model_id,
                        "Model_Type": MODEL_TYPE_MAP.get(model_id, "Unknown"),
                        "prompt_type": prompt_type,
                        "phase2_mode": args.phase2_mode,
                        "hernia_type_ground_truth": gt.get("ground_truth_hernia_type", ""),
                        "case_complexity": case_row.get("case_complexity", ""),
                        "run_timestamp": datetime.now().isoformat(),
                        "transcript_phase1_path": p1_transcript_path.as_posix(),
                        "transcript_phase2_path": p2_transcript_path.as_posix(),
                        "final_diagnosis": p1_scores.get("final_diagnosis", ""),
                        "correct_diagnosis": p1_scores.get("correct_diagnosis", ""),
                        "diagnosis_in_differential": p1_scores.get(
                            "diagnosis_in_differential", ""
                        ),
                        "questions_to_diagnosis": p1_scores.get(
                            "questions_to_final_diagnosis", ""
                        ),
                        "total_questions_asked": p1_scores.get("total_questions_asked", ""),
                        "relevant_questions": p1_scores.get("relevant_questions", ""),
                        "irrelevant_questions": p1_scores.get("irrelevant_questions", ""),
                        "redundant_questions": p1_scores.get("redundant_questions", ""),
                        "red_flag_pain_asked": p1_scores.get("red_flag_pain_asked", ""),
                        "red_flag_skin_changes_asked": p1_scores.get(
                            "red_flag_skin_changes_asked", ""
                        ),
                        "red_flag_obstructive_asked": p1_scores.get(
                            "red_flag_obstructive_asked", ""
                        ),
                        "red_flag_reducibility_asked": p1_scores.get(
                            "red_flag_reducibility_asked", ""
                        ),
                        "red_flag_systemically_unwell_asked": p1_scores.get(
                            "red_flag_systemically_unwell_asked", ""
                        ),
                        "missed_critical_red_flag": p1_scores.get(
                            "missed_critical_red_flag", ""
                        ),
                        "red_flag_coverage_score": rf_coverage,
                        "past_surgical_history_asked": p1_scores.get(
                            "past_surgical_history_asked", ""
                        ),
                        "anticoagulation_asked": p1_scores.get("anticoagulation_asked", ""),
                        "pregnancy_asked": p1_scores.get("pregnancy_asked", ""),
                        "comorbidities_asked": p1_scores.get("comorbidities_asked", ""),
                        "expressed_uncertainty": p1_scores.get("expressed_uncertainty", ""),
                        "overconfidence": p1_scores.get("overconfidence", ""),
                        "important_features_missed": p1_scores.get(
                            "important_features_missed", ""
                        ),
                        "appropriate_imaging": p2_scores.get(
                            "appropriate_imaging_recommended", ""
                        ),
                        "appropriate_referral": p2_scores.get("appropriate_referral", ""),
                        "urgency_correct": p2_scores.get("urgency_correct", ""),
                        "overtesting_present": p2_scores.get("overtesting_present", ""),
                        "overmanagement_present": p2_scores.get(
                            "overmanagement_present", ""
                        ),
                        "unsafe_recommendation": p2_scores.get(
                            "unsafe_recommendation", ""
                        ),
                        "severity_safety_issue": p2_scores.get(
                            "severity_safety_issue", ""
                        ),
                        "danger_score": danger_dict["danger_score"],
                        "danger_tier": danger_dict["danger_tier"],
                        "reviewer_id": "",
                        "overall_clinical_utility_score": "",
                        "empathy_score": "",
                        "actionability_score": "",
                        "readability_grade_level": "",
                        "reviewer_notes": "",
                    }

                    append_result_row(
                        result_row,
                        results_path,
                        write_header=not header_written["results"],
                    )
                    header_written["results"] = True

                    append_result_row(
                        danger_dict,
                        danger_path,
                        write_header=not header_written["danger"],
                    )
                    header_written["danger"] = True

                    log(
                        f"   Saved. Danger: {danger_dict['danger_score']} "
                        f"({danger_dict['danger_tier']})",
                        Fore.GREEN,
                    )

                except Exception as exc:
                    log(f"   Error: {exc}", Fore.RED)
                    error_row = {
                        "case_id": case_id,
                        "model": model_id,
                        "prompt_type": prompt_type,
                        "phase2_mode": args.phase2_mode,
                        "error": str(exc),
                        "run_timestamp": datetime.now().isoformat(),
                    }
                    append_result_row(
                        error_row,
                        errors_path,
                        write_header=False,
                    )

                time.sleep(RATE_LIMIT_DELAY)

    cost_tracker.save()
    cost_tracker.save_summary(str(cost_summary_path))

    log(f"\nPipeline complete: {run_count} runs finished", Fore.GREEN)
    log(f"Run directory: {run_dir.as_posix()}", Fore.GREEN)
    log(f"Results: {results_path.as_posix()}", Fore.GREEN)
    log(f"Danger scores: {danger_path.as_posix()}", Fore.GREEN)
    log(f"Cost summary: {cost_summary_path.as_posix()}\n", Fore.GREEN)

    return 0


if __name__ == "__main__":
    sys.exit(main())
