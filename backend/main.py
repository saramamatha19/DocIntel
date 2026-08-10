from fastapi import FastAPI, UploadFile, File
from pathlib import Path

from retriever import retrieve_documents
from call_llm import call_llm


app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
def home():
    return {"message" : "DocIntel Backend is Running!!!"}

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename  #upload/hr.pdf
    #with automatically handles closing the file
    with file_path.open("wb") as buffer:    #wb=write binary
        buffer.write(await file.read())

    return {
        "filename": file.filename,
        "message": "Document uploaded successfully"
    }