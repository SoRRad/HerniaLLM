# HerniaLLM

**Evaluation of LLM Performance in Ventral Hernia Clinical Decision-Making**

A fully automated pipeline to run clinical hernia cases through multiple LLMs, score outputs, track costs, and compute a Clinical Danger Score.

---

## What This Does

For each case in your CSV, this pipeline:
1. Runs a simulated patient–doctor conversation with each LLM
2. Tests three prompting strategies (zero / soft / hard)
3. Runs both Phase 1 (diagnosis) and Phase 2 (management)
4. Scores every output automatically
5. Computes a **Clinical Danger Score** per interaction
6. Tracks **API cost** per case, per model, per run
7. Saves full transcripts and a clean results CSV ready for statistical analysis

---

## One-Time Setup (Windows)

Follow these steps exactly, in order.

### Step 1 — Install Python

1. Go to https://www.python.org/downloads/
2. Download **Python 3.11** (click the yellow button)
3. Run the installer
4. **Important:** On the first screen, tick the box that says **"Add Python to PATH"** before clicking Install

Verify it worked — open **Command Prompt** (search "cmd" in Start menu) and type:
```
python --version
```
You should see `Python 3.11.x`

---

### Step 2 — Install VS Code

1. Go to https://code.visualstudio.com/
2. Download and install for Windows
3. Open VS Code → open the `HerniaLLM` folder (File → Open Folder)

---

### Step 3 — Install Claude Code

In VS Code, open the terminal (Terminal → New Terminal) and run:
```
npm install -g @anthropic-ai/claude-code
```

If `npm` is not found, install Node.js first from https://nodejs.org/ (LTS version), then repeat.

---

### Step 4 — Clone This Repo

In the VS Code terminal:
```
git clone https://github.com/SoRRad/HerniaLLM.git
cd HerniaLLM
```

---

### Step 5 — Install Python Packages

In the terminal, inside the HerniaLLM folder:
```
pip install -r requirements.txt
```

This installs everything the pipeline needs. It takes 1–2 minutes.

---

### Step 6 — Get Your API Keys

You need three API keys. All have free tiers to start.

#### OpenAI (GPT-4o)
1. Go to https://platform.openai.com/api-keys
2. Sign up / log in
3. Click **"Create new secret key"**
4. Copy the key (starts with `sk-...`)

#### Google (Gemini)
1. Go to https://aistudio.google.com/app/apikey
2. Sign in with Google account
3. Click **"Create API key"**
4. Copy the key

#### Anthropic (Claude)
1. Go to https://console.anthropic.com/
2. Sign up / log in
3. Go to API Keys → **"Create Key"**
4. Copy the key (starts with `sk-ant-...`)

#### NVIDIA (Nemotron Super)
Go to https://build.nvidia.com, sign in with a free account, click your profile → API Key → Generate Key. Free tier includes enough credits to run a full study pilot.

---

### Step 7 — Set Up Your API Keys

1. In the HerniaLLM folder, find the file called `.env.example`
2. Make a copy of it and rename the copy to `.env`
3. Open `.env` in VS Code and fill in your keys:

```
OPENAI_API_KEY=sk-...your key here...
GOOGLE_API_KEY=...your key here...
ANTHROPIC_API_KEY=sk-ant-...your key here...
```

**Never share your `.env` file or commit it to GitHub.** It is already listed in `.gitignore` for protection.

---

### Step 8 — Prepare Your Case Data

1. Open `data/cases_template.csv` in Excel
2. Fill in your de-identified hernia cases following the column guide
3. Save as `data/cases.csv`
4. Fill in `data/ground_truth.csv` with your pre-specified correct answers

See `data/COLUMN_GUIDE.md` for detailed instructions on each field.

---

## Supported Models

| Model | Provider | Type | API | Key Needed | Notes |
|-------|----------|------|-----|------------|-------|
| GPT-4o | OpenAI | LLM_Closed | Yes | OPENAI_API_KEY | |
| Gemini 1.5 Pro | Google | LLM_Closed | Yes | GOOGLE_API_KEY | |
| Claude Sonnet 4 | Anthropic | LLM_Closed | Yes | ANTHROPIC_API_KEY | |
| Llama Nemotron Super 49B | NVIDIA NIM | LLM_Open | Yes (OpenAI-compatible) | NVIDIA_API_KEY | Open weights, reproducible, free tier at build.nvidia.com |
| OpenEvidence | OpenEvidence | RAG | Manual | — | Manual entry only |
| Copilot | Microsoft | LLM_Closed_Manual | Manual | — | Manual entry only |

---

## Running the Pipeline

Open the VS Code terminal and run:

```bash
# Run all cases, all models, all prompt conditions
python pipeline/run_pipeline.py

# Run a single model only
python pipeline/run_pipeline.py --model gpt4o

# Run a specific prompt condition only
python pipeline/run_pipeline.py --prompt hard

# Run a quick test on the first 3 cases
python pipeline/run_pipeline.py --test
```

---

## Outputs

All outputs are saved to the `outputs/` folder:

| File | Contents |
|------|----------|
| `outputs/results/results.csv` | One row per case × model × prompt × phase — ready for SPSS/R/Excel |
| `outputs/results/cost_summary.csv` | API cost breakdown per model per run |
| `outputs/transcripts/` | Full conversation log for every interaction |
| `outputs/results/danger_scores.csv` | Clinical Danger Score per interaction |

---

## Using Claude Code

Claude Code lets you modify the pipeline using plain English. In the terminal:

```
claude
```

Then type what you want, for example:
- *"Add a new column to the output for overtesting"*
- *"Change the max questions per conversation to 15"*
- *"Show me the cost summary for the last run"*

---

## Project Structure

```
HerniaLLM/
├── README.md                  ← You are here
├── .env.example               ← API key template
├── .env                       ← Your keys (never committed)
├── .gitignore                 ← Protects sensitive files
├── requirements.txt           ← Python packages
│
├── data/
│   ├── cases_template.csv     ← Fill this with your cases
│   ├── cases.csv              ← Your populated case data
│   ├── ground_truth.csv       ← Pre-specified correct answers
│   └── COLUMN_GUIDE.md        ← Field-by-field instructions
│
├── pipeline/
│   ├── run_pipeline.py        ← Main entry point
│   ├── patient_sim.py         ← Patient simulator LLM
│   ├── models.py              ← API calls to each LLM
│   ├── scoring.py             ← Automated metric extraction
│   ├── danger_score.py        ← Clinical Danger Score calculator
│   └── cost_tracker.py        ← Token counting and cost logging
│
└── outputs/
    ├── transcripts/           ← Full conversation logs
    └── results/               ← Scored CSVs
```

---

## Citation

If you use this pipeline, please cite:
> [Your name], [Co-authors]. Evaluation of LLM Performance in Ventral Hernia Clinical Decision-Making. [Journal], [Year].

---

## Contact

GitHub: [@SoRRad](https://github.com/SoRRad)
# HerniaLLM
