import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import QueryRequest, QueryResponse, UrlIngestRequest, DeleteSourceRequest
from rag_engine import RAGEngine
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Gallery AI RAG API", description="API for Unity Integration")

# Allow all origins for the Wednesday Test
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RAGEngine()
UPLOAD_DIR = os.getenv("UPLOAD_DIRECTORY", "./data/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def health_check():
    return {"status": "online", "service": "Gallery AI"}

@app.post("/query", response_model=QueryResponse)
def ask_ai(request: QueryRequest):
    return QueryResponse(**dict(zip(["answer", "topic", "sources"], engine.query(request.question))))

@app.post("/admin/ingest-pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    file_location = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    try:
        engine.ingest_pdf(file_location)
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/ingest-url")
def ingest_url(request: UrlIngestRequest):
    try:
        engine.ingest_url(request.url)
        return {"status": "success", "url": request.url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/sources")
def get_sources():
    return {"sources": engine.list_sources()}

@app.delete("/admin/source")
def delete_source(request: DeleteSourceRequest):
    success, msg = engine.delete_source(request.source_path)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "success", "message": msg}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)