# Legal Contract Q&A Assistant

A RAG (Retrieval-Augmented Generation) system that answers questions about employment contracts using semantic search and local LLMs.

## What It Does

- Ingests multiple PDF employment contracts
- Uses semantic search to find relevant contract sections
- Generates natural language answers with source citations
- Runs entirely locally (no API keys required)

## Architecture
```
PDF Contracts
    ↓
PDF Loader (PyPDF)
    ↓
Text Chunking (1000 chars, 200 overlap)
    ↓
Embeddings (all-MiniLM-L6-v2)
    ↓
Vector Store (FAISS)
    ↓
Semantic Retrieval (top 3 chunks)
    ↓
LLM Generation (Llama 3.2 3B via Ollama)
    ↓
Answer + Sources
```

**Tech Stack:**
- **LLM**: Llama 3.2 3B (via Ollama)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **Framework**: LangChain
- **UI**: Gradio

## Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, runs on CPU)
- 4GB+ RAM

### Installation

1. **Install Ollama**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
```

2. **Clone and setup**
```bash
git clone https://github.com/LakshyaDuck/legal-contract-rag
cd legal-contract-rag
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Add contracts**
Place PDF employment contracts in the `contracts/` folder

4. **Run**
```bash
python app.py
```

Open http://localhost:7860 in your browser

## Example Queries

- "What is the termination clause?"
- "What are the confidentiality requirements?"
- "What is the notice period for resignation?"
- "Can the employee work for competitors after leaving?"

## Known Limitations

**Chunking Quality**: Fixed-size chunking (1000 chars) can split legal clauses mid-sentence, affecting retrieval accuracy. Future improvement: implement semantic chunking that respects clause boundaries.

**Multi-document synthesis**: Currently retrieves from individual contracts separately. Doesn't automatically compare clauses across multiple contracts.

**Domain specificity**: Trained on general employment contracts. Performance may vary on specialized agreements (NDAs, IP assignments, etc.).

## Future Improvements

1. **Better chunking**: Clause-aware splitting using legal document structure
2. **Re-ranking**: Add cross-encoder re-ranking for top-k results
3. **Comparison mode**: Highlight differences between similar clauses across contracts
4. **Fine-tuning**: Fine-tune embeddings on legal corpus for better retrieval

## Technical Learnings

- **Retrieval is 80% of RAG quality**: Better chunking/search matters more than larger LLMs
- **Vector store persistence**: Caching FAISS index reduces startup from 30s to 1s
- **Local-first design**: RTX 3050 GPU handles 3B parameter models fine for demos


## License

MIT

---

Built as part of learning RAG systems and transformer architectures. Feedback welcome!
