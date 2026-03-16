from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import gradio as gr
import os

# Load and process PDFs (one-time setup)
print("Loading contracts...")
documents = []
for filename in os.listdir("contracts"):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(f"contracts/{filename}")
        documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Larger to keep full clauses
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]  # Split on structure, not arbitrary length
)
chunks = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cuda'})
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Setup LLM
llm = Ollama(model="llama3.2:3b")

# Create prompt template
template = """Use the following context to answer the question about employment contracts.
If you cannot find the answer in the context, say so.

Context: {context}

Question: {question}

Answer:"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Build RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Gradio interface function
def ask_contract(question):
    if not question.strip():
        return "Please enter a question."

    try:
        answer = rag_chain.invoke(question)

        # Get source documents using invoke instead
        docs = retriever.invoke(question)
        sources = "\n\n---\n**Sources:**\n"
        for i, doc in enumerate(docs, 1):
            source_file = doc.metadata.get('source', 'Unknown').split('/')[-1]
            page = doc.metadata.get('page', 'Unknown')
            sources += f"\n{i}. {source_file} (Page {page})"

        return answer + sources
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
demo = gr.Interface(
    fn=ask_contract,
    inputs=gr.Textbox(
        label="Ask a question about the employment contracts",
        placeholder="e.g., What is the termination clause?",
        lines=2
    ),
    outputs=gr.Textbox(label="Answer", lines=10),
    title="Legal Contract Q&A Assistant",
    description="Ask questions about employment contracts. The system will search through uploaded contracts and provide answers with sources.",
    examples=[
        ["What is the termination clause?"],
        ["What are the confidentiality requirements?"],
        ["What is the notice period for resignation?"],
        ["What are the employee benefits mentioned?"]
    ],
    theme="soft"
)

print("Starting Gradio interface...")
demo.launch(share=False)

# After creating the vector store, add:
VECTOR_STORE_PATH = "faiss_index"

# Check if vector store exists
if os.path.exists(VECTOR_STORE_PATH):
    print("Loading existing vector store...")
    vectorstore = FAISS.load_local(
        VECTOR_STORE_PATH, 
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("Creating new vector store...")
    # Your existing code to load PDFs and create chunks
    documents = []
    for filename in os.listdir("contracts"):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(f"contracts/{filename}")
            documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save for next time
    print("Saving vector store...")
    vectorstore.save_local(VECTOR_STORE_PATH)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
