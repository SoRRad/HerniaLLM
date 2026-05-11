# HerniaLLM

**Evaluation of LLM Performance in Ventral Hernia Clinical Decision-Making**

HerniaLLM evaluates LLM and RAG performance in ventral hernia clinical decision-making using retrospective de-identified clinical scenarios.

The study has two phases:

1. Phase 1: iterative diagnostic reasoning. The model receives an opening patient statement and asks one question at a time until it reaches a diagnosis or hits the max-question limit.
2. Phase 2: confirmation and initial management planning. The model recommends confirmatory testing, referral, urgency, and initial management.

Prompting strategies are `zero`, `soft`, and `hard`.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local `.env` from `.env.example` and add only the API keys you need:

```bash
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...
NVIDIA_API_KEY=...
```

Never commit `.env`.

## Supported Models

These model IDs must match `pipeline/models.py`.

| CLI model ID | Provider |
|---|---|
| `gpt-4o` | OpenAI |
| `gemini-1.5-pro` | Google |
| `claude-sonnet-4-20250514` | Anthropic |
| `nemotron-super` | NVIDIA NIM |

## Before Any Real Run

1. Fill `data/cases.csv` from `data/cases_template.csv`.
2. Fill `data/ground_truth.csv` with adjudicated ground truth for every real case.
3. Confirm the data are de-identified.
4. Validate inputs before any API call:

```bash
python pipeline/validate_inputs.py
```

Recommended first-run workflow:

```bash
python pipeline/validate_inputs.py
python pipeline/run_pipeline.py --test --model gpt-4o --prompt hard
```

The pipeline also runs validation automatically before loading data or making API calls. If validation fails, the run stops.

## How To Run A One-Model Pilot

Run a single model and prompt type:

```bash
python pipeline/run_pipeline.py --test --model gpt-4o --prompt hard
```

Run a single model across all prompt types:

```bash
python pipeline/run_pipeline.py --model gpt-4o
```

Run all configured models and prompts:

```bash
python pipeline/run_pipeline.py
```

Choose how Phase 2 receives the diagnosis:

```bash
python pipeline/run_pipeline.py --model gpt-4o --prompt hard --phase2-mode model_dx
python pipeline/run_pipeline.py --model gpt-4o --prompt hard --phase2-mode ground_truth_dx
```

`model_dx` uses the model's own Phase 1 diagnosis. `ground_truth_dx` supplies `ground_truth_diagnosis` from `data/ground_truth.csv`.

## How To Find Outputs

Every run writes to a timestamped directory:

```text
outputs/runs/YYYYMMDD_HHMMSS/
  transcripts/
  results/results.csv
  results/danger_scores.csv
  results/cost_log.csv
  results/cost_summary.csv
  results/errors.csv
```

The latest run path is written to:

```text
outputs/latest_run.txt
```

## How To Export A Blinded Reviewer Packet

After a run completes:

```bash
python pipeline/export_reviewer_packet.py
```

By default this reads `outputs/latest_run.txt` and creates:

```text
outputs/runs/<run_id>/review/reviewer_packet.csv
```

Model names are blinded as `System_A`, `System_B`, `System_C`, etc. The export helper also creates blinded transcript copies under `review/transcripts/` so reviewer-facing paths do not reveal model names.

To export a specific run:

```bash
python pipeline/export_reviewer_packet.py --run-dir outputs/runs/YYYYMMDD_HHMMSS
```

## Safety Warning

Never commit:

- `.env`
- `data/cases.csv`
- `data/ground_truth.csv` after it contains real adjudicated data
- `data/ground_truth_real.csv`
- any PHI
- any file containing real patient data
- reviewer Excel workbooks or exports with patient-adjacent notes

`data/ground_truth.csv` is tracked only as a blank two-row template. Once it contains real study data, keep those changes local.

## Useful Checks

Run these before a pilot:

```bash
python pipeline/smoke_test.py
python pipeline/validate_inputs.py
python pipeline/run_pipeline.py --help
```

`smoke_test.py` does not call model APIs and does not require API keys.
