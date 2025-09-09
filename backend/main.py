from fastapi import FastAPI
import uvicorn 
from fastapi import UploadFile, File
from ingest import ingest_documents

app = FastAPI(title="Multi-Modal-RAG", description="Multi-Modal-RAG is a system that allows you to upload documents and query them using RAG.")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/documents/upload")
def upload_documents(file: UploadFile = File(...)):
    return ingest_documents(file)

@app.get("/documents/")
def list_documents():
    return {"status": "ok"}

@app.post("/query")
def query(query: str):
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run('main:app', host="0.0.0.0", port=8000)
