"""
models.py
Unified API wrappers for GPT-4o, Gemini 1.5 Pro, Claude Sonnet, and Nemotron Super.
Each wrapper returns (response_text, input_tokens, output_tokens).
"""

import os
from tenacity import retry, stop_after_attempt, wait_exponential


# ── Model identifiers ───────────────────────────────────────────────────────
MODEL_GPT4O        = "gpt-4o"
MODEL_GEMINI       = "gemini-1.5-pro"
MODEL_CLAUDE       = "claude-sonnet-4-20250514"
MODEL_NEMOTRON     = "nemotron-super"
MODEL_IDS          = [MODEL_GPT4O, MODEL_GEMINI, MODEL_CLAUDE, MODEL_NEMOTRON]

TEMPERATURE   = int(os.getenv("TEMPERATURE", 0))
MAX_TOKENS    = 1024


# ── GPT-4o ──────────────────────────────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def call_gpt4o(messages: list[dict]) -> tuple[str, int, int]:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=MODEL_GPT4O,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    text = response.choices[0].message.content.strip()
    usage = response.usage
    return text, usage.prompt_tokens, usage.completion_tokens


# ── Gemini 1.5 Pro ──────────────────────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def call_gemini(messages: list[dict]) -> tuple[str, int, int]:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(MODEL_GEMINI)

    # Convert OpenAI-style messages to Gemini format
    history = []
    for m in messages[:-1]:
        role = "user" if m["role"] == "user" else "model"
        history.append({"role": role, "parts": [m["content"]]})

    chat = model.start_chat(history=history)
    result = chat.send_message(
        messages[-1]["content"],
        generation_config=genai.types.GenerationConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
        )
    )
    text = result.text.strip()
    # Gemini token counts
    in_tok  = result.usage_metadata.prompt_token_count
    out_tok = result.usage_metadata.candidates_token_count
    return text, in_tok, out_tok


# ── Claude Sonnet ────────────────────────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def call_claude(messages: list[dict]) -> tuple[str, int, int]:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Separate system message if present
    system = ""
    chat_messages = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            chat_messages.append({"role": m["role"], "content": m["content"]})

    response = client.messages.create(
        model=MODEL_CLAUDE,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=chat_messages,
        temperature=TEMPERATURE,
    )
    text = response.content[0].text.strip()
    return text, response.usage.input_tokens, response.usage.output_tokens


# ── NVIDIA Nemotron Super 49B ────────────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def call_nemotron(messages: list[dict]) -> tuple[str, int, int]:
    from openai import OpenAI
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
    )
    response = client.chat.completions.create(
        model="nvidia/llama-3.3-nemotron-super-49b-v1",
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    text = response.choices[0].message.content.strip()
    usage = response.usage
    return text, usage.prompt_tokens, usage.completion_tokens


# ── Unified caller ───────────────────────────────────────────────────────────
def call_model(model_id: str, messages: list[dict]) -> tuple[str, int, int]:
    """
    Call any supported model by ID.
    Returns (response_text, input_tokens, output_tokens).
    """
    if model_id == MODEL_GPT4O:
        return call_gpt4o(messages)
    elif model_id == MODEL_GEMINI:
        return call_gemini(messages)
    elif model_id == MODEL_CLAUDE:
        return call_claude(messages)
    elif model_id == MODEL_NEMOTRON:
        return call_nemotron(messages)
    else:
        raise ValueError(f"Unknown model: {model_id}")
