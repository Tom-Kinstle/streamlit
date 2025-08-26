# pip install PyPDF2

import os
import glob
import re
import shutil
from pathlib import Path
from typing import List, Dict, Optional, TypedDict

import torch
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# PDF processing imports
import fitz  # PyMuPDF


from fastapi.middleware.cors import CORSMiddleware


# --------------------
# Data models
# --------------------
class QueryState(TypedDict):
    question: str
    hoa_name: str
    retrieved_chunks: List[Document]
    answer: str
    error: Optional[str]


class QueryRequest(BaseModel):
    question: str
    hoa_name: str
    k: int = 2


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    error: Optional[str] = None


class LoadHOARequest(BaseModel):
    hoa_name: str


class LoadHOAResponse(BaseModel):
    success: bool
    message: str
    hoa_name: str
    document_count: int = 0
    chunk_count: int = 0


# --------------------
# Main HOA Q&A System Class
# --------------------
class HOAQASystem:
    def __init__(self):
        """Initialize the HOA Q&A System."""
        self.load_environment()
        self.setup_directories()
        self.initialize_openai_client()
        
        # Global state
        self.vector_stores: Dict[str, Chroma] = {}
        self.embedding_model: Optional[HuggingFaceEmbeddings] = None
        
        # Configuration
        self.chunk_size = 1000
        self.chunk_overlap = 200

    def load_environment(self):
        """Load and validate environment variables."""
        load_dotenv()
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # For deployment, use a default data directory or allow environment override
        hoa_base_dir_env = os.getenv("HOA_BASE_DIR")
        if hoa_base_dir_env:
            self.hoa_base_dir = Path(hoa_base_dir_env).resolve()
        else:
            # Default to a data directory in the app folder
            self.hoa_base_dir = Path(__file__).parent / "data"
            self.hoa_base_dir.mkdir(parents=True, exist_ok=True)

    def setup_directories(self):
        """Setup required directories."""
        self.vectorstore_base_dir = Path(__file__).parent / "vectorstores"
        self.vectorstore_base_dir.mkdir(exist_ok=True)

    def initialize_openai_client(self):
        """Initialize the OpenAI client."""
        self.client = OpenAI(api_key=self.openai_api_key)

    def setup_embedding_model(self) -> HuggingFaceEmbeddings:
        """Setup and return the embedding model."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": device, "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 16}
        )

    def extract_pdf_text_pymupdf(self, file_path: str) -> str:
        """Extract text from PDF using PyMuPDF (fitz)."""
        try:
            text = ""
            doc = fitz.open(file_path)
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text += page.get_text() + "\n"
            doc.close()
            return text.strip()
        except Exception as e:
            print(f"PyMuPDF failed for {file_path}: {e}")
            return ""

    def extract_pdf_file(self, file_path: str) -> str:
        """Extract text content from a PDF file using multiple methods."""
        # Try PyMuPDF first (usually most reliable)
        text = self.extract_pdf_text_pymupdf(file_path)
        if text and len(text.strip()) > 100:  # Check if we got substantial content
            return text
        
        print(f"Failed to extract meaningful text from {file_path}")
        return ""

    def create_chunks(self, text: str, source_filename: str) -> List[Document]:
        """Create document chunks from text."""
        if not text.strip():
            return []
        
        # Clean up text
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\nARTICLE ", "\n\nSECTION ", "\n\nSec. ", "\n\n", "\n", ". ", " "],
            length_function=len,
            is_separator_regex=False
        )
        
        raw_docs = splitter.create_documents([text])
        return [
            Document(
                page_content=doc.page_content.strip(),
                metadata={"source_file": source_filename, "chunk_id": i}
            )
            for i, doc in enumerate(raw_docs)
        ]

    def get_available_hoas(self) -> List[str]:
        """Get list of available HOA folders."""
        hoa_folders = []
        for hoa_name in sorted(os.listdir(self.hoa_base_dir)):
            hoa_path = self.hoa_base_dir / hoa_name
            if hoa_path.is_dir():
                # Check if there are any PDF files in the directory
                has_pdf_files = bool(list(glob.glob(str(hoa_path / "**" / "*.pdf"), recursive=True)))
                if has_pdf_files:
                    hoa_folders.append(hoa_name)
        return hoa_folders

    def load_single_hoa(self, hoa_name: str) -> LoadHOAResponse:
        """Load a single HOA's PDF documents into vector store."""
        if hoa_name in self.vector_stores:
            return LoadHOAResponse(
                success=True,
                message=f"HOA '{hoa_name}' is already loaded.",
                hoa_name=hoa_name
            )
        
        hoa_path = self.hoa_base_dir / hoa_name
        if not hoa_path.is_dir():
            return LoadHOAResponse(
                success=False,
                message=f"HOA folder '{hoa_name}' not found.",
                hoa_name=hoa_name
            )

        # Find all PDF files
        pdf_files = []
        for pdf_path in glob.glob(str(hoa_path / "**" / "*.pdf"), recursive=True):
            print(f"Processing PDF: {pdf_path}")
            pdf_content = self.extract_pdf_file(pdf_path)
            if pdf_content:
                pdf_files.append({
                    "filename": os.path.basename(pdf_path), 
                    "text": pdf_content
                })
            else:
                print(f"Warning: Could not extract text from {pdf_path}")

        if not pdf_files:
            return LoadHOAResponse(
                success=False,
                message=f"No readable PDF files found for HOA: {hoa_name}",
                hoa_name=hoa_name
            )

        # Create chunks from all PDF files
        all_documents = []
        for pdf_file in pdf_files:
            chunks = self.create_chunks(pdf_file["text"], pdf_file["filename"])
            all_documents.extend(chunks)
            print(f"Created {len(chunks)} chunks from {pdf_file['filename']}")

        # Create vector store directory
        hoa_db_dir = self.vectorstore_base_dir / f"hoa_chroma_db_{hoa_name.lower().replace(' ', '_')}"
        
        # Try to load existing vector store first
        if hoa_db_dir.exists():
            try:
                vs = Chroma(
                    persist_directory=str(hoa_db_dir),
                    embedding_function=self.embedding_model
                )
                self.vector_stores[hoa_name] = vs
                print(f"Loaded existing HOA: {hoa_name}")
                
                return {
                    "success": True,
                    "message": f"Successfully loaded existing HOA: {hoa_name}",
                    "hoa_name": hoa_name,
                    "document_count": len(pdf_files),
                    "chunk_count": 0
                }
            except Exception as e:
                print(f"Could not load existing vector store: {e}")
                # Only remove if we absolutely have to
                pass

        try:
            vs = Chroma.from_documents(
                documents=all_documents,
                embedding=self.embedding_model,
                persist_directory=str(hoa_db_dir),
                collection_metadata={"hnsw:space": "cosine"}
            )
            self.vector_stores[hoa_name] = vs
            print(f"✅ Loaded HOA: {hoa_name} | PDFs: {len(pdf_files)} | Chunks: {len(all_documents)}")
            
            return LoadHOAResponse(
                success=True,
                message=f"Successfully loaded HOA: {hoa_name}",
                hoa_name=hoa_name,
                document_count=len(pdf_files),
                chunk_count=len(all_documents)
            )
        except Exception as e:
            return LoadHOAResponse(
                success=False,
                message=f"Error loading HOA '{hoa_name}': {str(e)}",
                hoa_name=hoa_name
            )

    def unload_hoa(self, hoa_name: str) -> dict:
        """Unload an HOA from memory."""
        if hoa_name in self.vector_stores:
            del self.vector_stores[hoa_name]
            return {"success": True, "message": f"HOA '{hoa_name}' unloaded successfully."}
        else:
            return {"success": False, "message": f"HOA '{hoa_name}' was not loaded."}

    def retrieve_and_answer(self, question: str, hoa_name: str, k: int) -> QueryResponse:
        """Retrieve relevant documents and generate an answer."""
        if hoa_name not in self.vector_stores:
            return QueryResponse(
                answer="", 
                sources=[], 
                error=f"HOA '{hoa_name}' not loaded. Please load it first using the 'Load HOA' button."
            )

        vector_store = self.vector_stores[hoa_name]
        results_with_scores = vector_store.similarity_search_with_score(question, k=k)
        retrieved_chunks = [doc for doc, _ in results_with_scores]

        if not retrieved_chunks:
            return QueryResponse(answer="No relevant information found.", sources=[], error=None)

        context = "\n\n".join(
            f"[Document: {doc.metadata.get('source_file')}]\n{doc.page_content}"
            for doc in retrieved_chunks
        )
        sources = list(set(doc.metadata.get('source_file', 'Unknown') for doc in retrieved_chunks))

        prompt = f"""
You are a compliance analyst reviewing HOA governing documents. 


QUESTION:
{question}

DOCUMENTS:
{context}

RESPONSE GUIDELINES:
- Use only the provided documents. 
- Keep responses concise and factual and do not infer or guess.
 - Always cite specific section references when available (e.g., "Section 7.4(b)", "Article III", "Section 2.1").
ANSWER:
"""

        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
                top_p=1
            )
            answer = completion.choices[0].message.content
            return QueryResponse(answer=answer, sources=sources, error=None)
        except Exception as e:
            return QueryResponse(answer="", sources=sources, error=str(e))

    def initialize_system(self):
        """Initialize the embedding model and system."""
        self.embedding_model = self.setup_embedding_model()
        print("HOA Q&A System started. Embedding model loaded.")
        print(f"HOA Base Directory: {self.hoa_base_dir}")
        print(f"Available HOAs: {len(self.get_available_hoas())}")

    def get_health_status(self) -> dict:
        """Get system health status."""
        return {
            "status": "healthy",
            "available_hoas": self.get_available_hoas(),
            "loaded_hoas": list(self.vector_stores.keys()),
            "embedding_model_loaded": self.embedding_model is not None,
            "base_dir": str(self.hoa_base_dir)
        }


# --------------------
# Initialize the system
# --------------------
hoa_system = HOAQASystem()

# --------------------
# FastAPI app
# --------------------
app = FastAPI(title="HOA Document Q&A", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    hoa_system.initialize_system()


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML frontend."""
    try:
        # Try to read the HTML file from the same directory as the script
        html_path = Path(__file__).parent / "index.html"
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text(encoding='utf-8'))
        else:
            # If index.html doesn't exist, return a simple message
            return HTMLResponse(content="""
            <html>
            <head><title>HOA Q&A System</title></head>
            <body>
                <h1>HOA Document Q&A System</h1>
                <p>Please create an index.html file in the same directory as your Python script.</p>
                <p>You can access the API directly at:</p>
                <ul>
                    <li><a href="/api/health">Health Check</a></li>
                    <li><a href="/api/hoas">Available HOAs</a></li>
                </ul>
            </body>
            </html>
            """)
    except Exception as e:
        return HTMLResponse(content=f"<html><body><h1>Error loading frontend</h1><p>{str(e)}</p></body></html>")


@app.get("/logo.jpg")
async def serve_logo():
    """Serve the logo image."""
    # Check for different possible filenames
    possible_names = ["logo.jpg", "logo.jpg.jpg", "logo.jpeg", "Pro_Elections_Logo_Color_Horiz_BL_TAG_500px@72ppi.jpg"]
    
    for filename in possible_names:
        logo_path = Path(__file__).parent / filename
        if logo_path.exists():
            print(f"Found logo at: {logo_path}")  # Debug print
            return FileResponse(logo_path, media_type="image/jpeg")
    
    # If not found, list what files are actually in the directory
    directory = Path(__file__).parent
    available_files = [f.name for f in directory.iterdir() if f.is_file()]
    print(f"Available files in directory: {available_files}")
    
    raise HTTPException(status_code=404, detail=f"Logo not found. Available files: {available_files}")


@app.get("/api/hoas")
async def get_hoas():
    """Get list of available HOAs."""
    return {
        "available_hoas": hoa_system.get_available_hoas(),
        "loaded_hoas": list(hoa_system.vector_stores.keys())
    }


@app.post("/api/load-hoa", response_model=LoadHOAResponse)
async def load_hoa(request: LoadHOARequest):
    """Load a specific HOA's documents."""
    return hoa_system.load_single_hoa(request.hoa_name)


@app.post("/api/query", response_model=QueryResponse)
async def query_docs(request: QueryRequest):
    """Query documents for a specific HOA."""
    return hoa_system.retrieve_and_answer(request.question, request.hoa_name, request.k)


@app.delete("/api/unload-hoa/{hoa_name}")
async def unload_hoa(hoa_name: str):
    """Unload an HOA from memory."""
    return hoa_system.unload_hoa(hoa_name)


@app.get("/api/health")
async def health_check():
    return hoa_system.get_health_status()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

    # conda activate rag_cleaned
# python code/rag_hoa/notebooks/fastapi_openai4.0.py
# http://localhost:8001