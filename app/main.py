from fastapi import FastAPI,UploadFile,File,HTTPException,Request
from pathlib import Path
import uuid, os
from app.model import UploadResponse,ChatRequest,ChatResponse
from app.rag_engine import process_pdf
from app.dependencies import active_session
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles




app=FastAPI(title="Temporary PDF Q&A Chatbot")

app.mount("/static", StaticFiles(directory="app/static"), name="static")




# your routes
@app.get("/", response_class=HTMLResponse)
async def ui():
    with open("app/static/index.html", "r", encoding="utf-8") as f:
        return f.read()


# Allow all origins (for local testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:8000"] for stricter
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR=Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/upload",response_model=UploadResponse)
async def upload_pdf(file:UploadFile=File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, detail="Only PDF files allowed")
    
    session_id=str(uuid.uuid4())[:8]
    file_path=UPLOAD_DIR/f"{session_id}_{file.filename}"

    with open(file_path,'wb') as f:
        f.write(await file.read())

    num_pages, num_chunks= await process_pdf(str(file_path),session_id)

    return UploadResponse(
        session_id=session_id,
        message="PDF uploaded and indexed temporarily",
        num_pages=num_pages,
        num_chunks=num_chunks
    )

from fastapi.responses import StreamingResponse

@app.post("/chat")
async def chat(request: ChatRequest):

    if request.session_id not in active_session:
        raise HTTPException(404,"Session expired")

    session = active_session[request.session_id]
    chain = session["chain"]

    async def generate():
        stream = chain.stream(
            {"input": request.message},
            config={"configurable":{"session_id":request.session_id}}
        )

        for chunk in stream:
            if "answer" in chunk:
                yield chunk["answer"]

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/sessions")
async def list_sessions():
    return {"active_sessions": list(active_session.keys())}

# Optional: Cleanup endpoint
@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in active_session:
        del active_session[session_id]
    return {"status": "deleted"}

# In your main.py or wherever you call uvicorn.run()
import uvicorn

if __name__ == "__main__":
    try:
        uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)  # your args
    except KeyboardInterrupt:
        pass  # Silently exit on Ctrl+C