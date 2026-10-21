from fastapi import FastAPI, UploadFile, File, HTTPException
import httpx
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
async def analyze_pcg(file: UploadFile = File(...)):

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

    # Temporary response
    # Later, this part will call Service 3 (audio-engine)
    # and Service 4 (agent-orchestrator)

    return {
        "status": "success",
        "filename": file.filename,
        "size_bytes": len(audio_data),
        "message": "WAV file received successfully",
        "audio_engine": "not connected yet",
        "agent_orchestrator": "not connected yet"
    }