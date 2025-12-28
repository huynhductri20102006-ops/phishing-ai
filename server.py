from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# =========================
# CONFIG
# =========================
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# =========================
# INIT APP
# =========================
app = FastAPI(title="Phishing Detection API")

# =========================
# LOAD MODEL (ONCE)
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()  # inference mode


# =========================
# REQUEST MODEL
# =========================
class PhishingRequest(BaseModel):
    text: str


# =========================
# RESPONSE MODEL
# =========================
class PhishingResponse(BaseModel):
    is_phishing: bool
    score: float


# =========================
# API ENDPOINT
# =========================
@app.post("/check", response_model=PhishingResponse)
def check_phishing(req: PhishingRequest):
    """
    Simple phishing detection using DistilBERT sentiment model.
    Output:
    - is_phishing: bool
    - score: confidence (0..1)
    """

    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

    # Label 1 = NEGATIVE (used as phishing signal)
    phishing_score = probs[0][1].item()

    return {
        "is_phishing": phishing_score > 0.7,
        "score": phishing_score
    }


# =========================
# ROOT CHECK
# =========================
@app.get("/")
def root():
    return {"status": "Phishing Detection API is running"}
