import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException

from audio_processing import preprocess_audio


app = FastAPI(
    title="CardioAgent-Ops Audio Engine",
    description="Service 3 - Cardiac audio preprocessing and inference engine",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "service": "audio-engine",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided"
        )

    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Only .wav files are accepted"
        )

    audio_bytes = await file.read()

    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="The WAV file is empty"
        )

    temp_path = None

    try:
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp_file:

            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        # Run the preprocessing pipeline
        result = preprocess_audio(temp_path)

        return {
            "status": "success",
            "filename": file.filename,
            "sample_rate": result["sample_rate"],
            "signal_length": len(result["signal"]),
            "mfcc_shape": list(result["mfcc"].shape),
            "message": "Audio preprocessing completed successfully",
            "prediction": "not implemented yet"
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Audio processing failed: {str(error)}"
        )

    finally:
        # Delete temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)