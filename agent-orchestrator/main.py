from contextlib import asynccontextmanager

import httpx
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import EXPERIMENT_NAME, LLM_MODEL, OLLAMA_URL, PROMPT_VERSION, RAG_URL, generate_report


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Traces go to this MLflow experiment. If MLflow is down the agent still
    # works, the traces are just lost.
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
    except Exception as error:
        print(f"MLflow unavailable, traces will not be saved: {error}")

    yield


app = FastAPI(
    title="CardioAgent-Ops Agent Orchestrator",
    description="Service 4 - LLM agent: audio prediction -> RAG (ESC guidelines) -> clinical note",
    version="1.0.0",
    lifespan=lifespan
)


class Prediction(BaseModel):
    """
    The "prediction" object returned by the audio-engine.
    """

    label: str
    probability_abnormal: float = Field(..., ge=0, le=1)
    model_name: str | None = None
    version: str | None = None
    model_type: str | None = None
    run_id: str | None = None


class ReportRequest(BaseModel):
    prediction: Prediction
    patient: str = Field("", description="Free text: age, sex, symptoms...")


def is_reachable(url: str):
    try:
        return httpx.get(url, timeout=3).status_code < 500
    except httpx.HTTPError:
        return False


@app.get("/")
def root():
    return {
        "service": "agent-orchestrator",
        "status": "running"
    }


@app.get("/health")
def health_check():
    llm = is_reachable(f"{OLLAMA_URL}/api/tags")
    rag = is_reachable(f"{RAG_URL}/health")

    return {
        "status": "healthy" if llm and rag else "degraded",
        "llm": "connected" if llm else "unreachable",
        "llm_model": LLM_MODEL,
        "rag_engine": "connected" if rag else "unreachable",
        "prompt_version": PROMPT_VERSION
    }


@app.post("/report")
def create_report(request: ReportRequest):
    """
    Turn an audio prediction into a clinical note grounded in the ESC guidelines.
    Every call is saved as a trace in MLflow (experiment "agent-orchestrator").
    """

    try:
        return generate_report(request.prediction.model_dump(), request.patient)
    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=503,
            detail=f"LLM or rag-engine unavailable: {str(error)}"
        )
