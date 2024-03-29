import os
from fastapi import FastAPI, File, UploadFile

from wisper_large.connection import large_whisper_model

app = FastAPI()


@app.post("/transcribe_audio/")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = None,
):
    model = large_whisper_model

    contents = await file.read()
    temp_dir = "temp"
    temp_file_path = f"{temp_dir}/{file.filename}"

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    with open(temp_file_path, "wb") as f:
        f.write(contents)

    result = model.transcribe(temp_file_path, language=language)
    transcription = result["text"]

    os.remove(temp_file_path)

    return {"transcription": transcription}
