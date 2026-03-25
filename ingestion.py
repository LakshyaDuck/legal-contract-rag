import logging
import shutil
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_documents(folder_path: Path) -> List[Document]:
    """Load all PDFs from a specific session folder."""
    documents = []
    if not folder_path.exists():
        logger.warning(f"Folder {folder_path} does not exist.")
        return documents

    for file_path in folder_path.iterdir():
        if file_path.suffix.lower() in config.ALLOWED_EXTENSIONS:
            try:
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
                # Add source metadata for citation
                for doc in docs:
                    doc.metadata["source_file"] = file_path.name
                documents.extend(docs)
                logger.info(f"Loaded: {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to load {file_path.name}: {e}")

    return documents


def create_vector_store(session_id: str) -> Optional[FAISS]:
    """
    Create a FAISS vector store for a specific session.
    Deletes existing index for this session to ensure fresh rebuild.
    """
    upload_path = config.get_session_upload_path(session_id)
    store_path = config.get_session_vectorstore_path(session_id)

    # 1. Clean existing index for this session
    if store_path.exists():
        logger.info(f"Clearing existing index for session {session_id}")
        shutil.rmtree(store_path)

    # 2. Load Documents
    logger.info(f"Loading documents from {upload_path}")
    documents = load_documents(upload_path)

    if not documents:
        logger.warning("No valid documents found. Skipping index creation.")
        return None

    # 3. Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks.")

    # 4. Create Embeddings (Force CPU)
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": config.EMBEDDING_DEVICE},
    )

    # 5. Create & Save Vector Store
    logger.info("Generating embeddings and saving FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(store_path))

    logger.info(f"Vector store saved to {store_path}")
    return vectorstore


def get_vector_store(session_id: str) -> Optional[FAISS]:
    """Load an existing FAISS index for a session."""
    store_path = config.get_session_vectorstore_path(session_id)

    if not store_path.exists():
        logger.warning(f"Vector store not found for session {session_id}")
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": config.EMBEDDING_DEVICE},
    )

    try:
        vectorstore = FAISS.load_local(
            str(store_path), embeddings, allow_dangerous_deserialization=True
        )
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        return None
