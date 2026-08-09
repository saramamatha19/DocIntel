from fastapi import FastAPI, UploadFile, File
from pathlib import Path

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
def home():
    return {"message" : "DocIntel Backend is Running!!!"}

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename

    with file_path.open("wb") as buffer:
        buffer.write(await file.read())

    return {
        "filename": file.filename,
        "message": "Document uploaded successfully"
    }