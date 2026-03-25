import logging
import time
from typing import List, Optional, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM

import config
import ingestion

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Prompt ---
TEMPLATE = """You are a legal assistant analyzing employment contracts.
Use ONLY the provided context to answer the question.
If the answer is not in the context, state clearly that you do not have enough information.

For every claim you make, you must cite the source file name in brackets like this: [Source: filename.pdf].

Context:
{context}

Question:
{question}

Answer:"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

# --- LLM Setup with timeout ---
llm = OllamaLLM(model=config.OLLAMA_MODEL, timeout=config.OLLAMA_TIMEOUT)


def format_docs(docs):
    """Combine documents into a single string with metadata."""
    return "\n\n".join(
        f"[File: {doc.metadata.get('source_file', 'Unknown')}, Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )


def get_rag_response(session_id: str, query: str, history: List[Tuple[str, str]]):
    """
    Main entry point for RAG inference.
    Returns: (answer_text, source_list_with_pages, response_time_seconds)
    """
    start_time = time.time()
    try:
        # 1. Load Vector Store
        vectorstore = ingestion.get_vector_store(session_id)
        if not vectorstore:
            return (
                "Not indexed any documents for this session yet. Please upload PDFs.",
                [],
                0.0,
            )

        # 2. Retrieve documents
        retriever = vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVAL_K})
        docs = retriever.invoke(query)

        # 3. Extract source information (file + page)
        source_pages = []
        for doc in docs:
            source_file = doc.metadata.get("source_file", "Unknown")
            page_num = doc.metadata.get("page", "?")
            source_pages.append(f"{source_file} (page {page_num})")
        unique_sources = list(
            dict.fromkeys(source_pages)
        )  # preserve order but remove duplicates

        # 4. Format context and run LLM chain
        context = format_docs(docs)
        rag_chain = (
            {"context": lambda _: context, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        answer = rag_chain.invoke(query)

        # 5. Log details
        elapsed = time.time() - start_time
        logger.info(
            f"Query for session {session_id} took {elapsed:.2f}s, retrieved {len(docs)} chunks from {len(unique_sources)} distinct sources."
        )
        logger.info(f"Sources: {unique_sources}")

        return answer, unique_sources, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"RAG Engine Error: {e}")
        return (
            f"Error processing request: {str(e)}. Ensure Ollama is running.",
            [],
            elapsed,
        )
