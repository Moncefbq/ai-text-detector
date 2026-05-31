from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from backend.api.detect import analyze_text

print("MAIN FILE LOADED")

app = FastAPI(
    title="AICheck Pro API",
    description="Advanced AI-generated text detection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    print("POST /detect CALLED")
    return analyze_text(request.text)
