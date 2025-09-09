from fastapi import UploadFile, File
from fastapi import HTTPException
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader,CSVLoader
from langchain_unstructured import UnstructuredLoader
import base64
import os
from langchain_cohere import CohereRerank
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


chroma_db = None
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
        print('file',file.filename)
        return ingest_image(file.filename)
    elif file_extension == "csv":
        return ingest_csv(file.filename)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

'''get the file url instead of file object'''
def ingest_pdf(file: UploadFile = File(...)):
    loader = PyMuPDFLoader(file.filename)
    documents = loader.load()
    for document in documents:
        metadata = document.metadata
        page_content = document.page_content

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents( page_content)
        chunks = [{"metadata": metadata, "page_content": chunk} for chunk in chunks]
        chroma_db = Chroma.from_documents(chunks, embedding=OpenAIEmbeddings())

    return "PDF ingested successfully"




def ingest_docx(file: UploadFile = File(...)):
    loader = Docx2txtLoader(file.filename)
    documents = loader.load()
    total_content = []
    for document in documents:
        metadata = document.metadata
        page_content = document.page_content
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(page_content)
        chunks = [{"metadata": metadata, "page_content": chunk} for chunk in chunks]
        chroma_db = Chroma.from_documents(chunks, embedding=OpenAIEmbeddings())

    return "DOCX ingested successfully"


def ingest_html(file: UploadFile = File(...)):
    loader = UnstructuredLoader(file.filename)
    documents = loader.load()
    total_content = []
    for document in documents:
        metadata = document.metadata
        page_content = document.page_content
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(page_content)
        chunks = [{"metadata": metadata, "page_content": chunk} for chunk in chunks]
        chroma_db = Chroma.from_documents(chunks, embedding=OpenAIEmbeddings())

    return "HTML ingested successfully"


def ingest_image(file: UploadFile = File(...)):

    image_data = base64.b64encode(file.file.read()).decode("utf-8")
    print('image_data',image_data)
    llm = init_chat_model("openai:gpt-4o-mini")

    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Describe the image:",
            },
            {
                "type": "image",
                "source_type": "base64",
                "data": image_data,
                "mime_type": "image/jpeg",
            },
        ],
    }
    response = llm.invoke([message])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(response.text())
    chroma_db = Chroma.from_documents(chunks, embedding=OpenAIEmbeddings())

    return "Image ingested successfully"



def ingest_csv(file: str):
    loader = CSVLoader(file.filename)
    documents = loader.load()
    total_content = []
    for document in documents:
        metadata = document.metadata
        page_content = document.page_content
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(page_content)
        chunks = [{"metadata": metadata, "page_content": chunk} for chunk in chunks]
        chroma_db = Chroma.from_documents(chunks, embedding=OpenAIEmbeddings())

    return "CSV ingested successfully"




def get_response(user_query: str):

    cohere_rank = CohereRerank(model="rerank-english-v3.0")
    response = chroma_db.similarity_search(user_query)
    response = cohere_rank.invoke(response)
    prompt = PromptTemplate(template="""You are a helpful assistant that can answer questions about the following text: {text}
    Question: {question}
    Answer: """, input_variables=["text", "question"])
    response = prompt.invoke({"text": response, "question": user_query})
    return response
