from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Load AI model
qa_pipeline = pipeline("text-generation", model="distilgpt2")

# Define request schema
class QuestionInput(BaseModel):
    question: str

@app.post("/ask")
async def answer_question(input_data: QuestionInput):
    response = qa_pipeline(input_data.question, max_length=100)
    return {"answer": response[0]["generated_text"]}
