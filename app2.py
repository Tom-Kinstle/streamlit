# HOA Document Q&A System - Updated Version with Mode Selection

import os
import glob
import re
import shutil
from pathlib import Path
from typing import List, Dict, Optional, TypedDict

import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# PDF processing imports
import fitz  # PyMuPDF

# Additional imports for RAG functionality
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import torch

# --------------------
# Data models
# --------------------
class QueryState(TypedDict):
    question: str
    hoa_name: str
    retrieved_chunks: List[Document]
    answer: str
    error: Optional[str]


# --------------------
# Main HOA Q&A System Class
# --------------------
class HOAQASystem:
    def __init__(self):
        """Initialize the HOA Q&A System."""
        self.load_environment()
        self.initialize_openai_client()
        self.setup_directories()
        
        # Global state - simple vector storage
        self.hoa_documents: Dict[str, List[Document]] = {}
        self.hoa_embeddings: Dict[str, np.ndarray] = {}
        self.embedding_model: Optional[SentenceTransformer] = None
        self.vector_stores: Dict[str, any] = {}
        
        # Configuration - Optimized for factual retrieval
        self.chunk_size = 1000
        self.chunk_overlap = 200

    def load_environment(self):
        """Load and validate environment variables."""
        if load_dotenv is not None:
            load_dotenv()
        
        # Try Streamlit secrets first, then environment variables  
        self.openai_api_key = None
        try:
            # Import should be at module level, but try here if needed
            self.openai_api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception as e:
            print(f"Streamlit secrets error: {e}")
            pass
            
        # Fallback to environment variable
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in Streamlit secrets or environment variables")
        
        # Documents are now stored locally in the git repo
        
        # For deployment, use a default data directory or allow environment override
        try:
            import streamlit as st
            hoa_base_dir_env = st.secrets.get("HOA_BASE_DIR")
        except:
            hoa_base_dir_env = None
            
        if not hoa_base_dir_env:
            hoa_base_dir_env = os.getenv("HOA_BASE_DIR")
            
        if hoa_base_dir_env:
            self.hoa_base_dir = Path(hoa_base_dir_env).resolve()
        else:
            # Default to a data directory in the app folder
            self.hoa_base_dir = Path(__file__).parent / "data"
            self.hoa_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Documents are stored in git repo - no download needed

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
        """Extract text content from a PDF file using PyMuPDF."""
        # Use PyMuPDF for text extraction
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
        if self.hoa_base_dir.exists():
            for hoa_name in sorted(os.listdir(self.hoa_base_dir)):
                hoa_path = self.hoa_base_dir / hoa_name
                if hoa_path.is_dir():
                    # Check if there are any PDF files in the directory
                    has_pdf_files = bool(list(glob.glob(str(hoa_path / "**" / "*.pdf"), recursive=True)))
                    if has_pdf_files:
                        hoa_folders.append(hoa_name)
        return hoa_folders

    def load_single_hoa(self, hoa_name: str) -> dict:
        """Load a single HOA's PDF documents into vector store."""
        if hoa_name in self.vector_stores:
            return {
                "success": True,
                "message": f"HOA '{hoa_name}' is already loaded.",
                "hoa_name": hoa_name,
                "document_count": 0,
                "chunk_count": 0
            }
        
        hoa_path = self.hoa_base_dir / hoa_name
        if not hoa_path.is_dir():
            return {
                "success": False,
                "message": f"HOA folder '{hoa_name}' not found.",
                "hoa_name": hoa_name,
                "document_count": 0,
                "chunk_count": 0
            }

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
            return {
                "success": False,
                "message": f"No readable PDF files found for HOA: {hoa_name}",
                "hoa_name": hoa_name,
                "document_count": 0,
                "chunk_count": 0
            }

        # Create chunks from all PDF files
        all_documents = []
        for pdf_file in pdf_files:
            chunks = self.create_chunks(pdf_file["text"], pdf_file["filename"])
            all_documents.extend(chunks)
            print(f"Created {len(chunks)} chunks from {pdf_file['filename']}")

        # Create vector store directory
        hoa_db_dir = self.vectorstore_base_dir / f"hoa_chroma_db_{hoa_name.lower().replace(' ', '_')}"
        if hoa_db_dir.exists():
            shutil.rmtree(hoa_db_dir)
        
        # Use persisted ChromaDB for better performance
        try:
            vs = Chroma.from_documents(
                documents=all_documents,
                embedding=self.embedding_model,
                persist_directory=str(hoa_db_dir),
                collection_metadata={"hnsw:space": "cosine"}
            )
            self.vector_stores[hoa_name] = vs
            print(f"Loaded HOA: {hoa_name} | PDFs: {len(pdf_files)} | Chunks: {len(all_documents)}")
            
            return {
                "success": True,
                "message": f"Successfully loaded HOA: {hoa_name}",
                "hoa_name": hoa_name,
                "document_count": len(pdf_files),
                "chunk_count": len(all_documents)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error loading HOA '{hoa_name}': {str(e)}",
                "hoa_name": hoa_name,
                "document_count": 0,
                "chunk_count": 0
            }

    def unload_hoa(self, hoa_name: str) -> dict:
        """Unload an HOA from memory."""
        if hoa_name in self.vector_stores:
            del self.vector_stores[hoa_name]
            return {"success": True, "message": f"HOA '{hoa_name}' unloaded successfully."}
        else:
            return {"success": False, "message": f"HOA '{hoa_name}' was not loaded."}

    def retrieve_and_answer(self, question: str, hoa_name: str, k: int, analysis_mode: str = "Quick Answer") -> dict:
        """Retrieve relevant documents and generate an answer."""
        if hoa_name not in self.vector_stores:
            return {
                "answer": "", 
                "sources": [], 
                "error": f"HOA '{hoa_name}' not loaded. Please load it first using the 'Load HOA' button."
            }

        vector_store = self.vector_stores[hoa_name]
        results_with_scores = vector_store.similarity_search_with_score(question, k=k)
        retrieved_chunks = [doc for doc, _ in results_with_scores]

        if not retrieved_chunks:
            return {"answer": "No relevant information found.", "sources": [], "error": None}

        context = "\n\n".join(
            f"[Document: {doc.metadata.get('source_file')}]\n{doc.page_content}"
            for doc in retrieved_chunks
        )
        sources = list(set(doc.metadata.get('source_file', 'Unknown') for doc in retrieved_chunks))

        if analysis_mode == "Quick Answer":
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
        else:  # Detailed Analysis
            prompt = f"""
You are a compliance analyst reviewing HOA governing documents. 

QUESTION:
{question}

DOCUMENTS:
{context}

MODE 2: Detailed Analysis

Expand the answer with:

Supporting details (quote exactly as written for numbers, dates, and percentages)

Citations (name the document and section/article when possible)

Distinctions between mandatory requirements ("must," "shall") vs. optional procedures ("may," "can")

Discrepancies if documents conflict

Limitations (state if information is missing, unclear, or only appears in the file name)

CRITICAL REQUIREMENTS:

If information is missing or unclear → say: "Not specified in the provided documents."

If query is blank, irrelevant, or not a question → say: "Please provide a question about homeowner association rules, procedures, and requirements."

If user asks for advice (e.g., "should we…," "is this a good idea") → say: "I cannot provide advice."

If a date is only in a file name and not in the text, you may use it only if no other date is given.

Keep responses under 200 words unless the question requires more detail.

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
            return {"answer": answer, "sources": sources, "error": None}
        except Exception as e:
            return {"answer": "", "sources": sources, "error": str(e)}

    def initialize_system(self):
        """Initialize the embedding model and system."""
        if self.embedding_model is None:
            self.embedding_model = self.setup_embedding_model()
            print("HOA Q&A System started. Embedding model loaded.")
            print(f"HOA Base Directory: {self.hoa_base_dir}")
            print(f"Available HOAs: {len(self.get_available_hoas())}")
        return True

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
# Streamlit App
# --------------------

def main():
    st.set_page_config(
        page_title="HOA Document Q&A",
        page_icon="🏠",
        layout="wide"
    )
    
    # Custom CSS for larger but reasonable text size
    st.markdown("""
    <style>
    .main .block-container {
        font-size: 1.2em;
    }
    .stSelectbox > div > div {
        font-size: 1.2em;
    }
    .stSelectbox select {
        font-size: 1.2em !important;
        height: auto !important;
        padding: 0.3em !important;
    }
    .stTextArea > div > div > textarea {
        font-size: 1.2em;
    }
    .stButton > button {
        font-size: 1.2em;
        height: auto;
        padding: 0.3em 0.6em;
    }
    .stMarkdown {
        font-size: 1.2em;
    }
    h1 {
        font-size: 1.8em !important;
    }
    h2, h3 {
        font-size: 1.4em !important;
    }
    .stInfo, .stSuccess, .stError, .stWarning {
        font-size: 1.2em;
    }
    </style>
    """, unsafe_allow_html=True)

    # Logo at the very top
    logo_path = Path(__file__).parent / "logo.jpg"
    if logo_path.exists():
        st.image(str(logo_path), width=750)

    # Initialize session state
    if 'hoa_system' not in st.session_state:
        try:
            st.session_state.hoa_system = HOAQASystem()
            st.session_state.system_initialized = False
            st.session_state.current_hoa = None
            st.session_state.chat_history = []
        except Exception as e:
            st.error(f"Error initializing system: {str(e)}")
            st.stop()

    # Initialize system if not done
    if not st.session_state.system_initialized:
        with st.spinner("🚀 Initializing HOA Q&A System..."):
            try:
                st.session_state.hoa_system.initialize_system()
                st.session_state.system_initialized = True
                st.success("✅ System initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize system: {str(e)}")
                st.stop()

    # Title and description
    st.title("🏠 HOA Document Q&A System")
    st.markdown("<p style='font-size: 1em;'>Select an HOA and ask questions about their governing documents.</p>", unsafe_allow_html=True)

    # Load available HOA directories
    available_hoas = st.session_state.hoa_system.get_available_hoas()
    if not available_hoas:
        st.info("No HOA directories found. Please upload your HOA documents to get started.")
        st.markdown("""
        **To use this system:**
        1. Create folders for each HOA in the data directory
        2. Place PDF documents in each HOA folder
        3. Refresh the page to see available HOAs
        """)
        return

    # HOA selection dropdown
    selected_hoa = st.selectbox(
        "Select HOA:",
        options=[""] + available_hoas,
        index=0 if not st.session_state.current_hoa else available_hoas.index(st.session_state.current_hoa) + 1,
        help="Choose which HOA's documents you want to query"
    )

    if not selected_hoa:
        st.info("Please select an HOA to begin.")
        return

    # Initialize RAG system if HOA selection changed
    if selected_hoa != st.session_state.current_hoa:
        st.session_state.current_hoa = selected_hoa
        st.session_state.chat_history = []

        with st.spinner(f"Loading {selected_hoa} documents..."):
            result = st.session_state.hoa_system.load_single_hoa(selected_hoa)
            if result["success"]:
                st.success(f"✅ {selected_hoa} documents loaded successfully!")
            else:
                st.error(f"❌ Failed to load {selected_hoa} documents: {result['message']}")
                return

    # Display current HOA context
    st.info(f"📁 Currently querying: **{selected_hoa}**")

    # Mode selection (minimal)
    analysis_mode = st.radio(
        "",
        options=["Quick Answer", "Detailed Analysis"],
        index=0,
        horizontal=True
    )

    # Display current question and answer
    if st.session_state.chat_history:
        # Show only the most recent Q&A
        q, a, sources = st.session_state.chat_history[-1]
        st.subheader("💬 Current Question & Answer")
        with st.container():
            st.markdown(f"**Question:** {q}")
            st.markdown(f"**Answer:** {a}")
            if sources:
                st.markdown(f"**Sources:** {', '.join(sources)}")
            st.divider()

    # Question input
    st.subheader("❓ Ask a Question")
    question = st.text_area(
        "Enter your question:",
        placeholder="e.g., What is the term of office for directors?",
        height=100,
        key="question_input"
    )

    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("🔍 Ask Question", type="primary", disabled=not question.strip()):
            with st.spinner("🤔 Analyzing documents..."):
                result = st.session_state.hoa_system.retrieve_and_answer(question, selected_hoa, 2, analysis_mode)
                if result["error"]:
                    st.error(f"❌ Error: {result['error']}")
                elif result["answer"]:
                    st.session_state.chat_history.append((question, result["answer"], result["sources"]))
                    st.rerun()
                else:
                    st.warning("⚠️ No answer could be generated.")

    with col2:
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()



if __name__ == "__main__":
    main()