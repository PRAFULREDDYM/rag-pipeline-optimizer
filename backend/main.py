# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import io

from .rag_pipelines import build_pipelines
from .evaluator import GroqEvaluator

try:
    import pypdf
except ImportError:
    pypdf = None

app = FastAPI(title="RAG Pipeline Optimizer")

# CORS for Streamlit / React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory state (good enough for student project)
PIPELINES = build_pipelines()
EVALUATOR = GroqEvaluator()
DOCUMENTS: List[str] = []


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class AskResponse(BaseModel):
    best_pipeline_id: str
    best_pipeline_name: str
    pipelines: List[Dict[str, Any]]


@app.post("/upload_docs")
async def upload_docs(files: List[UploadFile] = File(...)):
    """
    Upload HR policies or any docs.
    Supports .txt, .md, and .pdf (if pypdf installed).
    """
    global DOCUMENTS

    texts: List[str] = []
    for f in files:
        content = await f.read()
        if f.filename.lower().endswith((".txt", ".md")):
            texts.append(content.decode("utf-8", errors="ignore"))
        elif f.filename.lower().endswith(".pdf"):
            if not pypdf:
                raise HTTPException(
                    status_code=400,
                    detail="pypdf not installed; install pypdf to enable PDF support.",
                )
            reader = pypdf.PdfReader(io.BytesIO(content))
            txt = "\n".join([page.extract_text() or "" for page in reader.pages])
            texts.append(txt)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type for {f.filename}. Use .txt, .md, or .pdf",
            )

    if not texts:
        raise HTTPException(status_code=400, detail="No valid documents uploaded.")

    DOCUMENTS = texts

    # index for all pipelines
    for p in PIPELINES:
        p.index_documents(DOCUMENTS)

    return {"message": f"Indexed {len(DOCUMENTS)} documents in all pipelines."}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if not DOCUMENTS:
        raise HTTPException(
            status_code=400, detail="No documents uploaded yet. Call /upload_docs first."
        )

    pipeline_outputs = []
    for p in PIPELINES:
        out = p.answer(req.question, top_k=req.top_k)
        pipeline_outputs.append(out)

    evaluated = EVALUATOR.evaluate(req.question, pipeline_outputs)

    # pick best by overall score
    best = max(evaluated, key=lambda x: x["scores"]["overall"])

    return {
        "best_pipeline_id": best["pipeline_id"],
        "best_pipeline_name": best["pipeline_name"],
        "pipelines": evaluated,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}