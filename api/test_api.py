
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "api-gateway"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_analyze_pcg_valid_wav():
    response = client.post(
        "/analyze_pcg",
        files={"file": ("test.wav", b"fake wav content", "audio/wav")}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_analyze_pcg_invalid_extension():
    response = client.post(
        "/analyze_pcg",
        files={"file": ("test.mp3", b"audio content", "audio/mpeg")}
    )
    assert response.status_code == 400

def test_analyze_pcg_empty_file():
    response = client.post(
        "/analyze_pcg",
        files={"file": ("empty.wav", b"", "audio/wav")}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "The WAV file is empty"