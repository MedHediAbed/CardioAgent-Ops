import os

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import httpx


AUDIO_ENGINE_URL = os.getenv("AUDIO_ENGINE_URL", "http://localhost:8002")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8004")

# Create the FastAPI application
app = FastAPI(
    title="CardioAgent-Ops API Gateway",
    description="Service 5 - API Gateway for cardiac audio analysis",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "service": "api-gateway",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/analyze_pcg")
async def analyze_pcg(
    file: UploadFile = File(...),
    patient: str = Form("", description="Free text: age, sex, symptoms...")
):

    # Check that a file was provided
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided"
        )

    # Check that the uploaded file is a WAV file
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Only .wav files are accepted"
        )

    # Read the audio file
    audio_data = await file.read()

    # Check that the file is not empty
    if len(audio_data) == 0:
        raise HTTPException(
            status_code=400,
            detail="The WAV file is empty"
        )

    async with httpx.AsyncClient() as client:

        # Service 3 (audio-engine): preprocessing + heart sound classification
        try:
            response = await client.post(
                f"{AUDIO_ENGINE_URL}/analyze",
                files={"file": (file.filename, audio_data, "audio/wav")},
                timeout=60
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise HTTPException(
                status_code=502,
                detail=f"audio-engine failed: {str(error)}"
            )

        audio_result = response.json()
        prediction = audio_result["prediction"]

        # No model in the MLflow registry yet -> nothing to explain
        if not isinstance(prediction, dict):
            return {
                "status": "partial",
                "filename": file.filename,
                "audio_engine": audio_result,
                "agent_orchestrator": "skipped: no model available"
            }

        # Service 4 (agent-orchestrator): LLM + RAG clinical note
        try:
            response = await client.post(
                f"{ORCHESTRATOR_URL}/report",
                json={"prediction": prediction, "patient": patient},
                timeout=600   # the local LLM is slow on CPU
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise HTTPException(
                status_code=502,
                detail=f"agent-orchestrator failed: {str(error)}"
            )

    return {
        "status": "success",
        "filename": file.filename,
        "prediction": prediction,
        "report": response.json()
    }