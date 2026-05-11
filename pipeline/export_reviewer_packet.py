"""
Create a blinded reviewer packet from the latest HerniaLLM run.

By default this reads outputs/latest_run.txt, then writes:
  outputs/runs/<run_id>/review/reviewer_packet.csv

The packet never includes actual model names. Transcript files are copied into
a blinded review/transcripts folder so the file paths do not reveal model names.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


PACKET_COLUMNS = [
    "case_id",
    "prompt_type",
    "phase2_mode",
    "blinded_system_id",
    "transcript_phase1_path",
    "transcript_phase2_path",
    "final_diagnosis",
    "appropriate_imaging",
    "appropriate_referral",
    "urgency_correct",
    "unsafe_recommendation",
    "danger_score",
    "danger_tier",
    "reviewer_id",
    "reviewer_correct_diagnosis",
    "reviewer_question_relevance_score",
    "reviewer_management_appropriateness_score",
    "reviewer_safety_score",
    "reviewer_overall_clinical_utility_score",
    "reviewer_notes",
]


def safe_file_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def system_label(index: int) -> str:
    """Return System_A, System_B, ..., System_Z, System_AA, etc."""
    letters = []
    number = index + 1
    while number:
        number, remainder = divmod(number - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "System_" + "".join(reversed(letters))


def read_latest_run_path(latest_path: Path) -> Path:
    if not latest_path.exists():
        raise FileNotFoundError(
            f"{latest_path} was not found. Run the pipeline first, or pass --run-dir."
        )

    run_path_text = latest_path.read_text(encoding="utf-8").strip()
    if not run_path_text:
        raise ValueError(f"{latest_path} is empty. Run the pipeline again, or pass --run-dir.")

    return Path(run_path_text)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def infer_transcript_path(run_dir: Path, row: dict, phase: int) -> Path:
    case_id = row.get("case_id", "")
    model = row.get("model", "")
    prompt_type = row.get("prompt_type", "")
    filename = (
        f"{safe_file_part(case_id)}_{safe_file_part(model)}_"
        f"{safe_file_part(prompt_type)}_phase{phase}.json"
    )
    return run_dir / "transcripts" / filename


def resolve_transcript_path(run_dir: Path, row: dict, phase: int) -> Path:
    key = f"transcript_phase{phase}_path"
    stored_path = row.get(key, "")
    if stored_path:
        path = Path(stored_path)
        if path.exists():
            return path

    return infer_transcript_path(run_dir, row, phase)


def copy_blinded_transcript(
    source_path: Path,
    review_transcripts_dir: Path,
    row: dict,
    blinded_system_id: str,
    phase: int,
) -> str:
    if not source_path.exists():
        raise FileNotFoundError(f"Missing transcript file: {source_path}")

    filename = (
        f"{safe_file_part(row.get('case_id', 'case'))}_"
        f"{safe_file_part(blinded_system_id)}_"
        f"{safe_file_part(row.get('prompt_type', 'prompt'))}_"
        f"{safe_file_part(row.get('phase2_mode', 'model_dx'))}_phase{phase}.json"
    )
    blinded_path = review_transcripts_dir / filename
    shutil.copyfile(source_path, blinded_path)
    return blinded_path.as_posix()


def export_reviewer_packet(run_dir: Path) -> Path:
    results_path = run_dir / "results" / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Results file was not found: {results_path}")

    rows = read_csv(results_path)
    if not rows:
        raise ValueError(f"Results file has no rows to export: {results_path}")

    model_to_blinded_id: dict[str, str] = {}
    packet_rows: list[dict] = []
    review_dir = run_dir / "review"
    review_transcripts_dir = review_dir / "transcripts"
    review_transcripts_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        model = row.get("model", "")
        if model not in model_to_blinded_id:
            model_to_blinded_id[model] = system_label(len(model_to_blinded_id))
        blinded_system_id = model_to_blinded_id[model]

        phase1_source = resolve_transcript_path(run_dir, row, phase=1)
        phase2_source = resolve_transcript_path(run_dir, row, phase=2)

        packet_rows.append(
            {
                "case_id": row.get("case_id", ""),
                "prompt_type": row.get("prompt_type", ""),
                "phase2_mode": row.get("phase2_mode", ""),
                "blinded_system_id": blinded_system_id,
                "transcript_phase1_path": copy_blinded_transcript(
                    phase1_source, review_transcripts_dir, row, blinded_system_id, phase=1
                ),
                "transcript_phase2_path": copy_blinded_transcript(
                    phase2_source, review_transcripts_dir, row, blinded_system_id, phase=2
                ),
                "final_diagnosis": row.get("final_diagnosis", ""),
                "appropriate_imaging": row.get("appropriate_imaging", ""),
                "appropriate_referral": row.get("appropriate_referral", ""),
                "urgency_correct": row.get("urgency_correct", ""),
                "unsafe_recommendation": row.get("unsafe_recommendation", ""),
                "danger_score": row.get("danger_score", ""),
                "danger_tier": row.get("danger_tier", ""),
                "reviewer_id": "",
                "reviewer_correct_diagnosis": "",
                "reviewer_question_relevance_score": "",
                "reviewer_management_appropriateness_score": "",
                "reviewer_safety_score": "",
                "reviewer_overall_clinical_utility_score": "",
                "reviewer_notes": "",
            }
        )

    packet_path = review_dir / "reviewer_packet.csv"
    with packet_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PACKET_COLUMNS)
        writer.writeheader()
        writer.writerows(packet_rows)

    return packet_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a blinded reviewer packet from a HerniaLLM run."
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Run directory to export. Defaults to the path in outputs/latest_run.txt.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        run_dir = Path(args.run_dir) if args.run_dir else read_latest_run_path(
            Path("outputs") / "latest_run.txt"
        )
        packet_path = export_reviewer_packet(run_dir)
    except Exception as exc:
        print(f"Reviewer packet export failed: {exc}")
        return 1

    print("Reviewer packet created successfully.")
    print(f"Packet: {packet_path.as_posix()}")
    print("Model names were blinded as System_A, System_B, System_C, etc.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
