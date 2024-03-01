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
            {"role": "system", "content": "You are a story writing assistant."},
            {"role": "user", "content": question.question}
        ]
    )
    print("Response from Llama model:")
    answer = response['choices'][0]['message']['content']
    answer = answer.replace("\n[INST]", "")
    answer = answer.strip("\n")
    return {"answer": answer}
