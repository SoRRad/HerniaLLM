"""
Validate HerniaLLM input CSV files before any API calls are made.

This script is intentionally conservative and beginner-readable. It checks
that the case file and ground truth file line up before the pipeline spends
money or sends any clinical scenario text to an external model.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REQUIRED_CASE_COLUMNS = [
    "case_id",
    "hernia_type",
    "chief_complaint",
    "duration_symptoms",
    "location",
    "size_description",
    "reducible",
    "pain_character",
    "pain_severity",
    "aggravating_factors",
    "relieving_factors",
    "nausea_vomiting",
    "bowel_changes",
    "fever",
    "prior_hernia_repair",
    "prior_abdominal_surgery",
    "anticoagulation",
    "pregnancy",
    "bmi_category",
    "relevant_comorbidities",
    "red_flag_pain",
    "red_flag_skin_changes",
    "red_flag_obstructive_symptoms",
    "red_flag_irreducible",
    "red_flag_systemically_unwell",
    "case_complexity",
]

REQUIRED_GT_COLUMNS = [
    "case_id",
    "ground_truth_diagnosis",
    "ground_truth_hernia_type",
    "correct_imaging",
    "correct_referral",
    "referral_urgency",
    "red_flags_present",
    "critical_red_flags",
    "correct_management_summary",
    "adjudicator",
    "adjudication_date",
]

CASE_RED_FLAG_COLUMNS = [
    "red_flag_pain",
    "red_flag_skin_changes",
    "red_flag_obstructive_symptoms",
    "red_flag_irreducible",
    "red_flag_systemically_unwell",
]

GT_RED_FLAG_COLUMNS = ["red_flags_present"]

YES_NO_BLANK_NA = {"yes", "no", "", "na", "n/a"}
CASE_COMPLEXITY_VALUES = {"simple", "moderate", "complex", "very_complex", "", "na", "n/a"}
REFERRAL_URGENCY_VALUES = {"routine", "urgent", "emergency", "", "na", "n/a"}


def clean(value: object) -> str:
    """Return a stripped string for reliable CSV validation."""
    if value is None:
        return ""
    return str(value).strip()


def normalized(value: object) -> str:
    return clean(value).lower()


def read_csv_rows(path: Path, label: str, errors: list[str]) -> tuple[list[dict], list[str]]:
    """Read a CSV file and return rows plus headers, collecting readable errors."""
    if not path.exists():
        errors.append(f"{label} file was not found: {path}")
        return [], []

    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
    except Exception as exc:
        errors.append(f"{label} file could not be read as CSV: {path} ({exc})")
        return [], []

    if not fieldnames:
        errors.append(f"{label} file is empty or has no header row: {path}")

    return rows, fieldnames


def check_required_columns(
    fieldnames: list[str],
    required_columns: list[str],
    label: str,
    errors: list[str],
) -> None:
    missing = [col for col in required_columns if col not in fieldnames]
    if missing:
        errors.append(
            f"{label} file is missing required column(s): {', '.join(missing)}"
        )


def collect_case_ids(rows: list[dict], label: str, errors: list[str]) -> dict[str, dict]:
    """Validate non-empty and unique case_id values, then return rows by case_id."""
    rows_by_id: dict[str, dict] = {}
    seen_line_by_id: dict[str, int] = {}

    for line_number, row in enumerate(rows, start=2):
        case_id = clean(row.get("case_id"))
        if not case_id:
            errors.append(f"{label} row {line_number} has a blank case_id.")
            continue

        if case_id in rows_by_id:
            first_line = seen_line_by_id[case_id]
            errors.append(
                f"{label} has duplicate case_id '{case_id}' on rows "
                f"{first_line} and {line_number}."
            )
            continue

        rows_by_id[case_id] = row
        seen_line_by_id[case_id] = line_number

    return rows_by_id


def has_any_data(row: dict | None, columns: list[str]) -> bool:
    if not row:
        return False
    return any(clean(row.get(col)) for col in columns if col != "case_id")


def is_placeholder_pair(case_row: dict | None, gt_row: dict | None) -> bool:
    """A placeholder has only a case_id in both the case row and ground truth row."""
    return not has_any_data(case_row, REQUIRED_CASE_COLUMNS) and not has_any_data(
        gt_row, REQUIRED_GT_COLUMNS
    )


def check_allowed_values(
    rows_by_id: dict[str, dict],
    columns: list[str],
    allowed_values: set[str],
    label: str,
    errors: list[str],
    friendly_values: str,
) -> None:
    for case_id, row in rows_by_id.items():
        for col in columns:
            value = normalized(row.get(col))
            if value not in allowed_values:
                errors.append(
                    f"{label} row for {case_id}: '{col}' has value "
                    f"'{clean(row.get(col))}'. Use only {friendly_values}."
                )


def check_case_id_alignment(
    cases_by_id: dict[str, dict],
    gt_by_id: dict[str, dict],
    errors: list[str],
) -> None:
    case_ids = set(cases_by_id)
    gt_ids = set(gt_by_id)

    for case_id in sorted(case_ids - gt_ids):
        errors.append(
            f"cases.csv includes {case_id}, but ground_truth.csv has no matching row."
        )

    for case_id in sorted(gt_ids - case_ids):
        errors.append(
            f"ground_truth.csv includes {case_id}, but cases.csv has no matching row."
        )


def check_ground_truth_completeness(
    cases_by_id: dict[str, dict],
    gt_by_id: dict[str, dict],
    errors: list[str],
) -> None:
    """Require ground truth fields for real rows, while allowing blank templates."""
    fields_to_fill = [col for col in REQUIRED_GT_COLUMNS if col != "case_id"]

    for case_id in sorted(set(cases_by_id) & set(gt_by_id)):
        case_row = cases_by_id[case_id]
        gt_row = gt_by_id[case_id]

        if is_placeholder_pair(case_row, gt_row):
            continue

        for col in fields_to_fill:
            if clean(gt_row.get(col)) == "":
                errors.append(
                    f"Ground truth row for {case_id}: '{col}' is blank. "
                    "Fill it before a real run, or use NA if it truly does not apply."
                )


def validate_inputs(
    cases_path: str | Path = "data/cases.csv",
    gt_path: str | Path = "data/ground_truth.csv",
    verbose: bool = True,
) -> bool:
    """Return True when both CSV files are safe to use for a pipeline run."""
    cases_path = Path(cases_path)
    gt_path = Path(gt_path)
    errors: list[str] = []

    if verbose:
        print("HerniaLLM input validation")
        print(f"Cases file: {cases_path}")
        print(f"Ground truth file: {gt_path}")

    cases_exists = cases_path.exists()
    gt_exists = gt_path.exists()

    case_rows, case_columns = read_csv_rows(cases_path, "Cases", errors)
    gt_rows, gt_columns = read_csv_rows(gt_path, "Ground truth", errors)

    if cases_exists:
        check_required_columns(case_columns, REQUIRED_CASE_COLUMNS, "Cases", errors)
    if gt_exists:
        check_required_columns(gt_columns, REQUIRED_GT_COLUMNS, "Ground truth", errors)

    cases_by_id = collect_case_ids(case_rows, "Cases", errors) if cases_exists else {}
    gt_by_id = collect_case_ids(gt_rows, "Ground truth", errors) if gt_exists else {}

    if cases_exists and gt_exists:
        check_case_id_alignment(cases_by_id, gt_by_id, errors)

    if cases_exists:
        check_allowed_values(
            cases_by_id,
            CASE_RED_FLAG_COLUMNS,
            YES_NO_BLANK_NA,
            "Cases",
            errors,
            "yes, no, blank, or NA",
        )
        check_allowed_values(
            cases_by_id,
            ["case_complexity"],
            CASE_COMPLEXITY_VALUES,
            "Cases",
            errors,
            "simple, moderate, complex, very_complex, blank, or NA",
        )

    if gt_exists:
        check_allowed_values(
            gt_by_id,
            GT_RED_FLAG_COLUMNS,
            YES_NO_BLANK_NA,
            "Ground truth",
            errors,
            "yes, no, blank, or NA",
        )
        check_allowed_values(
            gt_by_id,
            ["referral_urgency"],
            REFERRAL_URGENCY_VALUES,
            "Ground truth",
            errors,
            "routine, urgent, emergency, blank, or NA",
        )

    if cases_exists and gt_exists:
        check_ground_truth_completeness(cases_by_id, gt_by_id, errors)

    if errors:
        if verbose:
            print("\nValidation failed. Please fix these issue(s) before running the pipeline:")
            for idx, error in enumerate(errors, start=1):
                print(f"{idx}. {error}")
            print("\nNo API calls were made.")
        return False

    if verbose:
        print("\nValidation passed.")
        print("The case file and ground truth file are aligned and ready for a pilot run.")

    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate HerniaLLM case and ground truth CSV files."
    )
    parser.add_argument("--cases", default="data/cases.csv", help="Path to cases CSV")
    parser.add_argument("--gt", default="data/ground_truth.csv", help="Path to ground truth CSV")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return 0 if validate_inputs(args.cases, args.gt) else 1


if __name__ == "__main__":
    sys.exit(main())
