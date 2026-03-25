import os
from pathlib import Path

# --- Project Paths ---
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
LOGS_DIR = BASE_DIR / "logs"

# --- Model Configuration ---
# For RTX 3050 6GB:
# Use llama3.2:3b for stability. Switch to qwen2.5:7b or if VRAM allows.
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_TIMEOUT = 60  # seconds
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"

# --- RAG Hyperparameters ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 3

# --- System Settings ---
ALLOWED_EXTENSIONS = {".pdf"}


def ensure_directories():
    """Create necessary directories if they don't exist."""
    for dir_path in [UPLOADS_DIR, VECTORSTORE_DIR, LOGS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


def get_session_upload_path(session_id: str) -> Path:
    return UPLOADS_DIR / session_id


def get_session_vectorstore_path(session_id: str) -> Path:
    return VECTORSTORE_DIR / session_id


# Initialize on import
ensure_directories()
