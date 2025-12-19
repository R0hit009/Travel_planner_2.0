from fastapi import FastAPI,HTTPException
# from rag_pipeline import rag_query
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import json
from main_ollama import search


app = FastAPI(title="TourPlanner AI Backend", version="1.0.0")
class BodyRequest(BaseModel):
    body: str

@app.get("/health")
async def health_check():
    return {"status": "ok"}


# ===== Fixed search endpoint =====
@app.post("/search")
async def stream_search(request: BodyRequest):

    async def event_stream():
        # yield tokens as they arrive
        for chunk in search(request.body):
            if chunk:
                data = json.dumps({"delta": chunk})
                yield f"data: {data}\n\n"
                
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
