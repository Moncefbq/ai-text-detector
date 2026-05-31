from fastapi import FastAPI
from pydantic import BaseModel
from backend.api.detect import analyze_text

app = FastAPI(
    title="AICheck Pro API",
    description="Advanced AI-generated text detection API",
    version="1.0.0"
)

class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {
        "message": "AICheck Pro API is running",
        "status": "online"
    }

@app.post("/detect")
def detect(request: TextRequest):
    return analyze_text(request.text)
