from fastapi import FastAPI
import uvicorn 
from fastapi import UploadFile, File
from ingest import ingest_documents
from ingest import get_response

app = FastAPI(title="Multi-Modal-RAG", description="Multi-Modal-RAG is a system that allows you to upload documents and query them using RAG.")
db=[]
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/documents/upload")
def upload_documents(file: UploadFile = File(...)):
    db.append(file)
    return ingest_documents(file)


@app.get("/documents/")
def list_documents():
    return db

@app.post("/query")
def query(query: str):
    return get_response(query)

if __name__ == "__main__":
    uvicorn.run('main:app', host="0.0.0.0", port=8000)
