from fastapi import UploadFile, File
from fastapi import HTTPException
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader,CSVLoader

def ingest_documents(file: UploadFile = File(...)):
    "get the file extension"
    file_extension = file.filename.split(".")[-1]
    if file_extension == "pdf":
        return ingest_pdf(file)
    elif file_extension == "docx":
        return ingest_docx(file.filename)
    elif file_extension == "html":
        return ingest_html(file.filename)
    elif file_extension == "jpg" or file_extension == "jpeg" or file_extension == "png":
        return ingest_image(file.filename)
    elif file_extension == "csv":
        return ingest_csv(file.filename)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

'''get the file url instead of file object'''
def ingest_pdf(file: UploadFile = File(...)):
    loader = PyMuPDFLoader(file.filename)
    documents = loader.load()
    return {"documents": documents}
    
def ingest_docx(file: UploadFile = File(...)):
    return "docx ingested"

def ingest_html(file: UploadFile = File(...)):
    return "html ingested"

def ingest_image(file: UploadFile = File(...)):
    return "image ingested"

def ingest_csv(file: str):
    return "csv ingested"