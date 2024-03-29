import os
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from typing import List

from paylina_wisper_tiny.connection import llm, tiny_whisper_model

app = FastAPI()


class Texts(BaseModel):
    texts: List[str]


class Prompt(BaseModel):
    prompt: str


class Question(BaseModel):
    question: str


def create_paolina_prompt(prompt, question: str):
    if not prompt:
        return f"""
    <<SYS>>
        You are Paolina, a very attractive 20-year-old, flirty, and playful adult model female obsessed with sex. Form detailed answers in a brutal sexual manner.
    <</SYS>>
    [INST]
        Question: {question}
    [/INST]\n
    Paolina:"""
    else:
        return f"""
    <<SYS>>
        {prompt}.
    <</SYS>>
    [INST]
        Question: {question}
    [/INST]\n
    Paolina:"""


@app.post("/generate_answer_text")
async def generate_answer_text(
    question: Question, prompt: str = None, max_tokens: int = None
):
    if prompt is None:
        paolina_prompt = create_paolina_prompt(prompt, question.question)
    else:
        paolina_prompt = create_paolina_prompt(prompt, question.question)
    if max_tokens is None:
        max_tokens = int(512)
    response = llm(
        paolina_prompt,
        max_tokens=max_tokens,
        stop=["[INST]", "None", "Question:"],
        echo=True,
    )
    full_response_text = response["choices"][0]["text"]
    answer = full_response_text.replace(paolina_prompt, "")
    return {"answer": answer}


@app.post("/embeddings-solar/")
async def get_embeddings(texts: Texts):
    embeddings = [llm.create_embedding(text) for text in texts.texts]
    print(len(embeddings[0]["data"][0]["embedding"]))
    return {"embeddings": embeddings}


@app.post("/transcribe_audio/")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = None,
):
    model = tiny_whisper_model

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
