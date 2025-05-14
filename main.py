from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import docx
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import logging
from pydantic import BaseModel
import db
import numpy as np
from fastapi.responses import JSONResponse
from fastapi.requests import Request

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ai-graderv3.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.options("/{rest_of_path:path}")
async def preflight_handler(request: Request, rest_of_path: str = ""):
    return JSONResponse(status_code=200, content={})

model = SentenceTransformer('all-MiniLM-L6-v2')
answer_key_text = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_docx(file_path: str):
    doc = docx.Document(file_path)
    return '\n'.join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(file_path: str):
    with pdfplumber.open(file_path) as pdf:
        return ''.join([page.extract_text() for page in pdf.pages if page.extract_text()])

def compare_with_answer_key(extracted_text: str, answer_key: str, sentence_similarity_threshold: float = 0.7):
    extracted_embedding = model.encode(extracted_text, convert_to_tensor=True)
    answer_key_embedding = model.encode(answer_key, convert_to_tensor=True)
    cosine_similarity = util.pytorch_cos_sim(extracted_embedding, answer_key_embedding)

    # New Part: Semantic matching (Improved with original naming)
    answer_key_sentences = [s.strip() for s in answer_key.split('.') if s.strip()]
    extracted_sentences = [s.strip() for s in extracted_text.split('.') if s.strip()]

    matched_points = []
    missing_points = []
    max_similarities = []

    if answer_key_sentences and extracted_sentences:
        answer_key_embeddings = model.encode(answer_key_sentences, convert_to_tensor=True)
        extracted_embeddings = model.encode(extracted_sentences, convert_to_tensor=True)

        for i, answer_embedding in enumerate(answer_key_embeddings):
            max_similarity = -1
            similarities = util.pytorch_cos_sim(answer_embedding, extracted_embeddings)[0].cpu().numpy()
            max_similarity = np.max(similarities)

            max_similarities.append(max_similarity)

            if max_similarity >= sentence_similarity_threshold:
                matched_points.append(answer_key_sentences[i])
            else:
                missing_points.append(answer_key_sentences[i])
    elif answer_key_sentences:
        missing_points.extend(answer_key_sentences)
        max_similarities.extend([0.0] * len(answer_key_sentences))

    overall_semantic_coverage = np.mean(max_similarities) if max_similarities else 0.0

    return {
        "similarity_score": cosine_similarity.item(),
        "overall_semantic_coverage": overall_semantic_coverage,
        "message": "Comparison complete with semantic point analysis.",
        "matched_points": matched_points,
        "missing_points": missing_points
    }

@app.post("/api/upload_answer_key/")
async def upload_answer_key(file: UploadFile = File(...)):
    global answer_key_text
    os.makedirs("uploads", exist_ok=True)
    file_location = f"uploads/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file.filename.endswith('.docx'):
        answer_key_text = extract_text_from_docx(file_location)
    elif file.filename.endswith('.pdf'):
        answer_key_text = extract_text_from_pdf(file_location)
    else:
        return {"error": "Unsupported file type. Please upload a .docx or .pdf file."}

    return {"filename": file.filename, "message": "Answer key uploaded successfully!"}

@app.post("/api/upload/")
async def upload_file(file: UploadFile = File(...), user_id: str = Form(...)):
    if not answer_key_text:
        return {"error": "Answer key is not uploaded yet."}

    os.makedirs("uploads", exist_ok=True)
    file_location = f"uploads/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file.filename.endswith('.docx'):
        extracted_text = extract_text_from_docx(file_location)
    elif file.filename.endswith('.pdf'):
        extracted_text = extract_text_from_pdf(file_location)
    else:
        return {"error": "Unsupported file type."}

    comparison_result = compare_with_answer_key(extracted_text, answer_key_text)
    db.store_comparison(user_id, file.filename, comparison_result["similarity_score"])

    return {
        "filename": file.filename,
        "extracted_text": extracted_text,
        "comparison_result": comparison_result
    }

@app.get("/api/comparisons/{user_id}")
def get_user_comparisons(user_id: str):
    return db.get_user_comparisons(user_id)

class User(BaseModel):
    username: str
    password: str

@app.post("/api/register")
def register(user: User):
    success = db.register_user(user.username, user.password)
    if not success:
        raise HTTPException(status_code=400, detail="Username already exists")
    return {"message": "User registered successfully"}

@app.post("/api/login")
def login(user: User):
    user_id = db.login_user(user.username, user.password)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"message": "Login successful", "user_id": user_id}
