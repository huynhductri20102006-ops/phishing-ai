from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

class EmailRequest(BaseModel):
    text: str

class PhishingResponse(BaseModel):
    is_phishing: bool
    score: float

@app.post("/check", response_model=PhishingResponse)
def check_phishing(req: EmailRequest):
    result = classifier(req.text)[0]
    label = result["label"]
    score = float(result["score"])

    return {
        "is_phishing": label == "NEGATIVE",
        "score": score
    }
