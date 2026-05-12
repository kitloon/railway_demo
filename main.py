import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from models import QueryRequest, QueryResponse, UrlIngestRequest, DeleteSourceRequest
from rag_engine import RAGEngine
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI application
app = FastAPI(title="Gallery AI RAG API", description="API for Unity Integration")

# Configure CORS to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the RAG Engine and setup upload directory
engine = RAGEngine()
UPLOAD_DIR = os.getenv("UPLOAD_DIRECTORY", "./data/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Health check endpoint to verify service status
@app.get("/")
def health_check():
    return {"status": "online", "service": "Gallery AI"}

# Standard query endpoint (Synchronous)
@app.post("/query", response_model=QueryResponse)
def ask_ai(request: QueryRequest):
    # Pass the question to the engine and return answer, topic, and sources
    answer, topic, sources = engine.query(request.question)
    return QueryResponse(answer=answer, topic=topic, sources=sources)

# Streaming query endpoint for real-time text generation
@app.post("/query_stream")
def ask_ai_stream(request: QueryRequest):
    # Returns a streaming response for better user experience
    return StreamingResponse(engine.stream_query(request.question), media_type="text/plain")

# Admin endpoint to upload and process PDF documents
@app.post("/admin/ingest-pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    # Save the uploaded file to the local directory
    file_location = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    try:
        # Feed the saved PDF to the RAG engine for vectorization
        count = engine.ingest_pdf(file_location)
        return {"status": "success", "filename": file.filename, "chunks_ingested": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Admin endpoint to scrape and ingest content from a URL
@app.post("/admin/ingest-url")
def ingest_url(request: UrlIngestRequest):
    try:
        # Process the URL directly using the RAG engine
        count = engine.ingest_url(request.url)
        return {"status": "success", "url": request.url, "chunks_ingested": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Admin endpoint to retrieve a list of all current knowledge sources
@app.get("/admin/sources")
def get_sources():
    # Fetches unique source paths from the vector database
    return {"sources": engine.list_sources()}

# Admin endpoint to delete a specific source from the knowledge base
@app.delete("/admin/source")
def delete_source(request: DeleteSourceRequest):
    # Remove all related vectors/chunks for the specified source
    success, msg = engine.delete_source(request.source_path)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "success", "message": msg}

# Start the server using Uvicorn
if __name__ == "__main__":
    import uvicorn
    # Listen on all network interfaces on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)