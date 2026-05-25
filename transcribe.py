import whisperx
import torch
import os
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# SETTINGS
# -----------------------------

AUDIO_FILE = "audio/CPA-Meeting-May22-2026.m4a"

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found. Check your .env file.")

MODEL_SIZE = "medium"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8

COMPUTE_TYPE = "float16"

# -----------------------------
# LOAD WHISPER MODEL
# -----------------------------

print("Loading WhisperX model...")

model = whisperx.load_model(
    MODEL_SIZE,
    DEVICE,
    compute_type=COMPUTE_TYPE
)

# -----------------------------
# LOAD AUDIO
# -----------------------------

print("Loading audio...")

audio = whisperx.load_audio(AUDIO_FILE)

# -----------------------------
# TRANSCRIBE
# -----------------------------

print("Transcribing audio...")

result = model.transcribe(
    audio,
    batch_size=BATCH_SIZE,
    print_progress=True
)

# -----------------------------
# ALIGNMENT
# -----------------------------

print("Aligning transcript...")

model_a, metadata = whisperx.load_align_model(
    language_code=result["language"],
    device=DEVICE
)

result = whisperx.align(
    result["segments"],
    model_a,
    metadata,
    audio,
    DEVICE,
    print_progress=True
)
# -----------------------------
# SAVE INTERMEDIATE TRANSCRIPT BEFORE DIARIZATION
# -----------------------------

os.makedirs("output", exist_ok=True)

intermediate_file = "output/transcript_before_diarization.txt"

print(f"Saving intermediate transcript to: {intermediate_file}")

with open(intermediate_file, "w", encoding="utf-8") as f:
    for segment in result["segments"]:
        start = round(segment["start"], 2)
        end = round(segment["end"], 2)
        text = segment["text"]

        f.write(f"[{start}s -> {end}s] {text}\n")

print("Intermediate transcript saved.")

# -----------------------------
# SPEAKER DIARIZATION
# -----------------------------

print("Running speaker diarization...")

diarize_model = whisperx.diarize.DiarizationPipeline(
    token=HF_TOKEN,
    device=DEVICE
)

diarize_segments = diarize_model(
    AUDIO_FILE,
    min_speakers=3,
    max_speakers=3
)

print("Diarization output:")
print(diarize_segments)
print(type(diarize_segments))

print("Assigning speakers to transcript...")

result = whisperx.assign_word_speakers(
    diarize_segments,
    result
)

print("Speaker assignment complete.")

# -----------------------------
# SAVE OUTPUT
# -----------------------------

os.makedirs("output", exist_ok=True)

output_file = "output/transcript.txt"

print(f"Saving transcript to: {output_file}")

with open(output_file, "w", encoding="utf-8") as f:

    for segment in result["segments"]:

        speaker = segment.get("speaker", "UNKNOWN")

        text = segment["text"]

        start = round(segment["start"], 2)

        end = round(segment["end"], 2)

        f.write(
            f"[{start}s -> {end}s] {speaker}: {text}\n"
        )

print("Done.")