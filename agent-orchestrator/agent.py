import os

import httpx
import mlflow
from mlflow.entities import SpanType


RAG_URL = os.getenv("RAG_URL", "http://localhost:8003")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
TOP_K = int(os.getenv("TOP_K", "4"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300"))   # CPU inference is slow

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "agent-orchestrator")

# Bump when a prompt below changes: every MLflow trace is tagged with it,
# so reports produced by different prompts can be compared
PROMPT_VERSION = "v1"

QUERY_PROMPT = """You help a cardiologist search the ESC (European Society of Cardiology) guidelines.

Heart sound (PCG) analysis: {label} (probability of abnormal: {probability:.0%})
Patient information: {patient}

Write ONE search query (at most 20 words) to find the guideline recommendations
on what to do next for this patient (examinations, follow-up).
Answer with the query only, no explanation."""

REPORT_SYSTEM = """You are a cardiology assistant writing a pre-consultation note for a physician.
Use ONLY the ESC guideline excerpts given. Cite every recommendation as [source, p. X].
If the excerpts do not cover a point, say so. Never state a definitive diagnosis."""

REPORT_PROMPT = """Heart sound (PCG) analysis by an AI model ({model}): {label},
probability of abnormal: {probability:.0%}.
Patient information: {patient}

ESC guideline excerpts:
{passages}

Write a short note with these sections:
1. Findings
2. Recommended next examinations (with citations)
3. Limits of this automatic analysis"""


@mlflow.trace(span_type=SpanType.LLM)
def call_llm(messages: list[dict]):
    """
    One chat completion with the local LLM (Ollama).
    """

    response = httpx.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2}
        },
        timeout=LLM_TIMEOUT
    )
    response.raise_for_status()
    body = response.json()

    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({
            "model": LLM_MODEL,
            "prompt_tokens": body.get("prompt_eval_count"),
            "completion_tokens": body.get("eval_count")
        })

    return body["message"]["content"].strip()


@mlflow.trace(span_type=SpanType.CHAIN)
def write_search_query(prediction: dict, patient: str):
    """
    Step 1 - the LLM turns the audio result into a question for the RAG engine.
    """

    prompt = QUERY_PROMPT.format(
        label=prediction["label"],
        probability=prediction["probability_abnormal"],
        patient=patient or "not provided"
    )

    return call_llm([{"role": "user", "content": prompt}]).strip('"')


@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve(query: str, top_k: int = TOP_K):
    """
    Step 2 - ESC guideline passages from the rag-engine.
    """

    response = httpx.post(
        f"{RAG_URL}/search",
        json={"query": query, "top_k": top_k},
        timeout=60
    )
    response.raise_for_status()

    return response.json()["results"]


def format_passages(passages: list[dict]):
    return "\n\n".join(
        f"[{p['source']}, p. {p['page']}]\n{p['text']}"
        for p in passages
    )


@mlflow.trace(span_type=SpanType.CHAIN)
def write_report(prediction: dict, patient: str, passages: list[dict]):
    """
    Step 3 - the LLM writes the note, grounded in the retrieved passages.
    """

    prompt = REPORT_PROMPT.format(
        model=prediction.get("model_type") or "heart-sound classifier",
        label=prediction["label"],
        probability=prediction["probability_abnormal"],
        patient=patient or "not provided",
        passages=format_passages(passages) or "(no excerpt found)"
    )

    return call_llm([
        {"role": "system", "content": REPORT_SYSTEM},
        {"role": "user", "content": prompt}
    ])


@mlflow.trace(name="cardio_report", span_type=SpanType.AGENT)
def generate_report(prediction: dict, patient: str = ""):
    """
    Complete agent: audio prediction -> search query -> RAG -> report.
    The whole run is one MLflow trace, linked to the audio model version.
    """

    mlflow.update_current_trace(tags={
        "prompt_version": PROMPT_VERSION,
        "llm_model": LLM_MODEL,
        "audio_model": str(prediction.get("model_name")),
        "audio_model_version": str(prediction.get("version")),
        "audio_model_type": str(prediction.get("model_type")),
        "audio_run_id": str(prediction.get("run_id"))
    })

    query = write_search_query(prediction, patient)
    passages = retrieve(query)
    report = write_report(prediction, patient, passages)

    return {
        "query": query,
        "passages": passages,
        "report": report,
        "trace_id": mlflow.get_active_trace_id()
    }
