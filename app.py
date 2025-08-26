# HOA Document Q&A System - Deployment Version

import os
import glob
import re
import shutil
from pathlib import Path
from typing import List, Dict, Optional, TypedDict

import torch
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# PDF processing imports
import pdfplumber
import gdown
import requests
from urllib.parse import urlparse
import zipfile


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
        
        # Google Drive folder ID for HOA documents
        self.google_drive_folder_id = "1m8Kaak_kN5ewMxWMUJqByq06QJPl_5Ql"
        
        # For deployment, use a default data directory or allow environment override
        hoa_base_dir_env = os.getenv("HOA_BASE_DIR")
        if hoa_base_dir_env:
            self.hoa_base_dir = Path(hoa_base_dir_env).resolve()
        else:
            # Default to a data directory in the app folder
            self.hoa_base_dir = Path(__file__).parent / "data" / "hoa_documents"
            self.hoa_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Download documents from Google Drive if directory is empty
        self.ensure_documents_downloaded()

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

    def ensure_documents_downloaded(self):
        """Download HOA documents from Google Drive if not present locally."""
        if self.hoa_base_dir.exists() and any(self.hoa_base_dir.iterdir()):
            print("HOA documents found locally, skipping download.")
            return
        
        print("Downloading HOA documents from Google Drive...")
        try:
            # Create temporary download directory
            temp_dir = Path(__file__).parent / "temp_download"
            temp_dir.mkdir(exist_ok=True)
            
            # Download the entire folder as zip
            zip_url = f"https://drive.google.com/uc?id={self.google_drive_folder_id}&export=download"
            
            # Use requests for the download with session handling
            session = requests.Session()
            response = session.get(zip_url, stream=True)
            
            if response.status_code == 200:
                zip_path = temp_dir / "hoa_documents.zip"
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Extract the zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Move extracted content to the HOA base directory
                extracted_folders = [d for d in temp_dir.iterdir() if d.is_dir()]
                if extracted_folders:
                    # If there's a main folder, move its contents
                    main_folder = extracted_folders[0]
                    if main_folder.name != "hoa_documents":
                        for item in main_folder.iterdir():
                            if item.is_dir():
                                shutil.move(str(item), str(self.hoa_base_dir))
                    else:
                        # Move the hoa_documents folder contents
                        for item in main_folder.iterdir():
                            if item.is_dir():
                                shutil.move(str(item), str(self.hoa_base_dir))
                
                # Clean up
                shutil.rmtree(temp_dir)
                print("HOA documents downloaded and extracted successfully!")
                
            else:
                print(f"Failed to download documents. Status code: {response.status_code}")
                
        except Exception as e:
            print(f"Error downloading HOA documents: {e}")
            # Continue without documents - app will show appropriate message

    def extract_pdf_text_pdfplumber(self, file_path: str) -> str:
        """Extract text from PDF using pdfplumber."""
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
        except Exception as e:
            print(f"pdfplumber failed for {file_path}: {e}")
            return ""

    def extract_pdf_file(self, file_path: str) -> str:
        """Extract text content from a PDF file using pdfplumber."""
        # Use pdfplumber for text extraction
        text = self.extract_pdf_text_pdfplumber(file_path)
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

    def retrieve_and_answer(self, question: str, hoa_name: str, k: int) -> dict:
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

        prompt = f"""
You are a compliance analyst reviewing HOA governing documents. 
Answer the question strictly using the provided documents.

QUESTION:
{question}

DOCUMENTS:
{context}

RESPONSE GUIDELINES:
- Use only the provided documents. Keep responses concise and factual and do not infer or guess.

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
                result = st.session_state.hoa_system.retrieve_and_answer(question, selected_hoa, 2)
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

    # Example questions
    st.subheader("💡 Example Questions")
    examples = [
        "Number of Directors?",
        "Term of Office?",
        "Candidate Qualifications/Eligibility Requirements?",
        "Are write-ins allowed?",
        "Is there a process for handling tie votes?"
    ]
    
    for example in examples:
        if st.button(f"📝 {example}", key=f"example_{example}"):
            # Set the question and trigger processing directly
            with st.spinner("🤔 Analyzing documents..."):
                result = st.session_state.hoa_system.retrieve_and_answer(example, selected_hoa, 2)
                if result["error"]:
                    st.error(f"❌ Error: {result['error']}")
                elif result["answer"]:
                    st.session_state.chat_history = [(example, result["answer"], result["sources"])]
                    st.rerun()
                else:
                    st.warning("⚠️ No answer could be generated.")


if __name__ == "__main__":
    main()

