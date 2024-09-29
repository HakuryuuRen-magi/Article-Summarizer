from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import article_summarize  # Assuming article_summarize function is defined in main.py

# FastAPI instance
app = FastAPI()

# Input data model
class SummarizeRequest(BaseModel):
    text: str

# Route to summarize text
@app.post("/summarize")
def summarize_text(request: SummarizeRequest):
    try:
        # Call the Summarize function
        summary = article_summarize(request.text)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root route to check API status
@app.get("/")
def read_root():
    return {"message": "FastAPI Summarizer API is running!"}
