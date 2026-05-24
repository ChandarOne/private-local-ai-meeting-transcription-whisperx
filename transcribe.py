import whisperx
import torch
import os

# -----------------------------
# SETTINGS
# -----------------------------

AUDIO_FILE = "audio/sample.m4a"

HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_SIZE = "medium"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8

COMPUTE_TYPE = "float32"

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
    batch_size=BATCH_SIZE
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
    DEVICE
)

# -----------------------------
# SPEAKER DIARIZATION
# -----------------------------

print("Running speaker diarization...")

diarize_model = whisperx.diarize.DiarizationPipeline(
    token=HF_TOKEN,
    device=DEVICE
)

diarize_segments = diarize_model(audio)

result = whisperx.assign_word_speakers(
    diarize_segments,
    result
)

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