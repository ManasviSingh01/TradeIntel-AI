from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from .rag import engine
from .schemas import QueryRequest, QueryResponse, DocumentResponse

app = FastAPI(
    title="TradeIntel AI",
    description="RAG-based commodity intelligence API for grounded Q&A over trade documents.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATA = Path(__file__).resolve().parents[1] / "data"

@app.on_event("startup")
def load_seed_documents():
    for path in DATA.glob("*.txt"):
        engine.add_document(path.name, path.read_text(encoding="utf-8"))

@app.get("/")
def root():
    return {"project": "TradeIntel AI", "status": "running", "documents": len({c['document'] for c in engine.chunks})}

@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": len(engine.chunks)}

@app.post("/documents", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    raw = await file.read()
    if suffix == ".txt":
        text = raw.decode("utf-8", errors="ignore")
    elif suffix == ".pdf":
        import io
        reader = PdfReader(io.BytesIO(raw))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        raise HTTPException(400, "Upload a .txt or .pdf document.")
    added = engine.add_document(file.filename or "document", text)
    return {"filename": file.filename or "document", "chunks_added": added}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    contexts = engine.retrieve(req.question, req.top_k)
    answer, mode = engine.answer(req.question, contexts)
    sources = [
        {"document": c["document"], "score": round(c["score"], 4), "excerpt": c["text"][:260]}
        for c in contexts
    ]
    return {"answer": answer, "sources": sources, "mode": mode}
