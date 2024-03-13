from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from connection import llm
from pprint import pprint


app = FastAPI()


class Question(BaseModel):
    question: str


def create_paolina_prompt(question: str):
    return f"""
You are Paolina, a very attractive 20-year-old, flirty, and playful adult model female obsessed with sex. Form detailed answers in a brutal sexual manner.
Question: {question}
Answer:"""


@app.post("/generate_answer_text")
async def generate_answer_text(question: Question):
    paolina_prompt = create_paolina_prompt(question.question)
    response = llm(paolina_prompt, max_tokens=256, stop=["</s>"], echo=True)
    print("Response from Llama model:")
    print(response)
    full_response_text = response['choices'][0]['text']

    # Looking for the start of the answer by finding the "Answer:" part
    answer_start_index = full_response_text.find("Answer:") + len("Answer:")
    if answer_start_index > -1:
        # Extract everything after "Answer:"
        answer_text = full_response_text[answer_start_index:].strip()
    else:
        # If "Answer:" is not found, fall back to the full response or an error message
        answer_text = "Could not extract the answer."

    pprint(answer_text)
    return {"answer": answer_text}


