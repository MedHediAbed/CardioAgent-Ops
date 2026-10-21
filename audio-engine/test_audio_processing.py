from audio_processing import preprocess_audio


FILE_PATH = "test.wav"

result = preprocess_audio(FILE_PATH)

print("Sample rate:", result["sample_rate"])
print("Signal length:", len(result["signal"]))
print("MFCC shape:", result["mfcc"].shape)