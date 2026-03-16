from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# Load PDFs
documents = []
for filename in os.listdir("contracts"):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(f"contracts/{filename}")
        documents.extend(loader.load())

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Setup LLM
llm = Ollama(model="llama3.2:3b")

# Create prompt template
template = """Use the following context to answer the question.
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

# Test query
query = "What is the termination clause?"
answer = rag_chain.invoke(query)
print(f"Answer: {answer}")