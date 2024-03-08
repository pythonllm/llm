from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from connection import llm

app = FastAPI()


class Question(BaseModel):
    question: str


@app.post("/generate_answer_text")
def generate_answer_text(question: Question):
    response = llm.create_chat_completion(
        messages=[
            {"role": "You are Paolina, a very attractive 20-year-old, flirty, and playful adult model female obsessed with sex. Form detailed answers in a brutal sexual manner"},
            {"role": "user", "content": question.question}
        ]
    )
    print("Response from Llama model:")
    answer = response['choices'][0]['message']['content']
    answer = answer.replace("\n[INST]", "")
    answer = answer.strip("\n")
    return {"answer": answer}
