# src/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import numpy as np
import logging
import warnings
from sentence_transformers import SentenceTransformer
from together import Together

# Suppress benign warnings from transformers
warnings.filterwarnings('ignore', message='.*position_ids.*')
warnings.filterwarnings('ignore', category=UserWarning)

from .prompts import FINAL_PROMPT_TEMPLATE
from .config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VECTOR_STORE = Config.VECTOR_STORE
EMBEDDING_MODEL = Config.EMBEDDING_MODEL
LLM_MODEL = Config.LLM_MODEL
MAX_CONTEXT_DOCS = Config.MAX_CONTEXT_DOCS
TOGETHER_API_KEY = Config.TOGETHER_API_KEY

app = FastAPI(title="Enterprise RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for web UI
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"Static files mounted from: {static_dir}")

class Query(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str
    sources: list

# Resolve VECTOR_STORE path
VECTOR_STORE_PATH = VECTOR_STORE
if not os.path.isabs(VECTOR_STORE_PATH):
    VECTOR_STORE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), VECTOR_STORE_PATH))

logger.info(f"Looking for vector store at: {VECTOR_STORE_PATH}")

# Load vector store
if not os.path.exists(VECTOR_STORE_PATH):
    logger.error(f"❌ Vector store not found at {VECTOR_STORE_PATH}")
    logger.error("Run: python -m src.ingest")
    raise RuntimeError(f"Vector store not found. Run `python -m src.ingest` first")

try:
    data = np.load(VECTOR_STORE_PATH, allow_pickle=True)
    embeddings = data["embeddings"]
    docs = data["docs"]
    metadatas = data["metadatas"]
    logger.info(f"✅ Vector store loaded: {len(docs)} documents")
except Exception as e:
    logger.error(f"❌ Error loading vector store: {e}")
    raise

# Load embedding model
try:
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("✅ Embedding model loaded")
except Exception as e:
    logger.error(f"❌ Error loading embedding model: {e}")
    raise

# Initialize Together client
if not TOGETHER_API_KEY:
    logger.error("❌ TOGETHER_API_KEY not set in .env")
    raise RuntimeError("TOGETHER_API_KEY not configured. Check .env file")

try:
    client = Together(api_key=TOGETHER_API_KEY)
    logger.info("✅ Together AI client initialized")
except Exception as e:
    logger.error(f"❌ Error initializing Together client: {e}")
    raise

def cosine_similarity(a, b):
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(b_norm, a_norm)

def retrieve_top_k(question, k=3):
    q_emb = embed_model.encode(question, convert_to_numpy=True)
    sims = cosine_similarity(q_emb, embeddings)
    idx = np.argsort(-sims)[:k]
    results = []
    for i in idx:
        results.append({
            "score": float(sims[i]),
            "text": str(docs[i]),
            "metadata": dict(metadatas[i])
        })
    return results

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "docs_count": len(docs),
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL
    }

@app.post("/chat", response_model=Answer)
async def chat(q: Query):
    """Chat endpoint for RAG queries"""
    question = q.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    retrieved = retrieve_top_k(question, k=2)  # Reduced from 3 to 2 docs for speed
    context_blocks = []
    for r in retrieved:
        src = r["metadata"].get("source", "unknown")
        excerpt = r["text"][:400] + ("..." if len(r["text"]) > 400 else "")  # Reduced from 800 to 400
        context_blocks.append(f"[{src}]\n{excerpt}")

    context = "\n\n".join(context_blocks) if context_blocks else "No retrieved documents."
    final_prompt = FINAL_PROMPT_TEMPLATE.format(context=context, question=question)

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an enterprise-grade RAG assistant. Follow citations and privacy rules."},
                {"role": "user", "content": final_prompt}
            ],
            max_tokens=300,  # Reduced from 800 to 300 for faster response
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM request failed: {str(e)}")

    if not context_blocks:
        text += "\n\n[Note: No retrieved sources found; answer based on model reasoning]"

    return Answer(answer=text, sources=[r["metadata"] for r in retrieved])

@app.get("/")
async def root():
    """Root endpoint - serves the web UI"""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    
    return {
        "message": "Enterprise RAG Chatbot API",
        "status": "healthy",
        "web_ui": "Open http://localhost:8000 to access the web interface"
    }
