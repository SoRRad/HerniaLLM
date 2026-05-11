"""
Smoke tests for a safe HerniaLLM pilot setup.

This script does not call external model APIs and does not require API keys.
It checks that the repo can load, templates exist, and the pipeline help text
is available before anyone starts a real run.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = PROJECT_ROOT / "pipeline"

REQUIRED_TEMPLATE_FILES = [
    PROJECT_ROOT / "data" / "cases_template.csv",
    PROJECT_ROOT / "data" / "ground_truth.csv",
]

REQUIRED_MODULES = [
    "csv",
    "json",
    "argparse",
    "dotenv",
    "colorama",
    "tqdm",
    "tenacity",
    "openai",
    "google.generativeai",
    "anthropic",
]

LOCAL_MODULES = [
    "validate_inputs",
    "models",
    "patient_sim",
    "scoring",
    "danger_score",
    "cost_tracker",
]

SECRET_MARKERS = [
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "NVIDIA_API_KEY",
    "sk-",
    "sk_ant",
]


def pass_msg(message: str) -> None:
    print(f"PASS: {message}")


def fail_msg(message: str) -> None:
    print(f"FAIL: {message}")


def check_templates() -> list[str]:
    errors: list[str] = []
    for path in REQUIRED_TEMPLATE_FILES:
        if path.exists():
            pass_msg(f"Template exists: {path.relative_to(PROJECT_ROOT)}")
        else:
            errors.append(f"Missing template file: {path.relative_to(PROJECT_ROOT)}")
    return errors


def check_imports() -> list[str]:
    errors: list[str] = []
    for module_name in REQUIRED_MODULES:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                importlib.import_module(module_name)
            pass_msg(f"Python module imports: {module_name}")
        except Exception as exc:
            errors.append(f"Could not import Python module '{module_name}': {exc}")

    sys.path.insert(0, str(PIPELINE_DIR))
    for module_name in LOCAL_MODULES:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                importlib.import_module(module_name)
            pass_msg(f"Pipeline module imports: {module_name}")
        except Exception as exc:
            errors.append(f"Could not import pipeline module '{module_name}': {exc}")

    return errors


def check_run_pipeline_help() -> tuple[list[str], str]:
    errors: list[str] = []
    command = [sys.executable, "pipeline/run_pipeline.py", "--help"]
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        return [f"Could not run pipeline help command: {exc}"], ""

    combined_output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        errors.append(
            "pipeline/run_pipeline.py --help did not exit cleanly. "
            f"Exit code: {completed.returncode}"
        )
    else:
        pass_msg("run_pipeline.py help command loads successfully")

    return errors, combined_output


def check_env_not_exposed(command_output: str) -> list[str]:
    errors: list[str] = []
    if ".env" in command_output:
        errors.append("The help command printed '.env'. It should not expose env file details.")

    for marker in SECRET_MARKERS:
        if marker in command_output:
            errors.append(
                f"The help command printed secret-looking marker '{marker}'. "
                "Do not expose API key names or values in smoke test output."
            )

    gitignore_path = PROJECT_ROOT / ".gitignore"
    if gitignore_path.exists():
        gitignore_text = gitignore_path.read_text(encoding="utf-8", errors="replace")
        if ".env" in gitignore_text:
            pass_msg(".env is listed in .gitignore")
        else:
            errors.append(".env is not listed in .gitignore")
    else:
        errors.append(".gitignore was not found")

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        pass_msg(".env exists locally, but this smoke test did not read or print it")
    else:
        pass_msg("No local .env file was found")

    if not errors:
        pass_msg("No .env contents or API-key-looking values were exposed")

    return errors


def main() -> int:
    print("HerniaLLM smoke test")
    print("This check does not call any model APIs and does not need API keys.\n")

    errors: list[str] = []
    errors.extend(check_templates())
    errors.extend(check_imports())
    help_errors, help_output = check_run_pipeline_help()
    errors.extend(help_errors)
    errors.extend(check_env_not_exposed(help_output))

    if errors:
        print("\nSmoke test failed. Please fix these issue(s):")
        for idx, error in enumerate(errors, start=1):
            fail_msg(f"{idx}. {error}")
        return 1

    print("\nSmoke test passed. The repo is ready for input validation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
