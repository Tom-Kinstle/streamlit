# Simplified HOA Document Q&A System for Streamlit Cloud
import os
import glob
import re
from pathlib import Path
from typing import List, Dict, Optional
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import fitz  # PyMuPDF

class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class SimpleHOASystem:
    def __init__(self):
        # Get API key from Streamlit secrets
        try:
            self.openai_api_key = st.secrets["OPENAI_API_KEY"]
        except:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            st.error("OpenAI API key not found in secrets or environment")
            st.stop()
            
        self.client = OpenAI(api_key=self.openai_api_key)
        self.hoa_base_dir = Path(__file__).parent / "data"
        
        # Simple storage
        self.hoa_documents: Dict[str, List[Document]] = {}
        self.hoa_embeddings: Dict[str, np.ndarray] = {}
        self.embedding_model = None

    @st.cache_resource
    def get_embedding_model(_self):
        return SentenceTransformer('all-MiniLM-L6-v2')  # Smaller, faster model

    def extract_pdf_text(self, file_path: str) -> str:
        try:
            text = ""
            doc = fitz.open(file_path)
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text += page.get_text() + "\n"
            doc.close()
            return text.strip()
        except Exception as e:
            st.error(f"PDF extraction failed for {file_path}: {e}")
            return ""

    def create_chunks(self, text: str, source_filename: str, chunk_size: int = 1000) -> List[Document]:
        if not text.strip():
            return []
        
        # Simple chunking by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(Document(
                        page_content=current_chunk.strip(),
                        metadata={"source_file": source_filename, "chunk_id": len(chunks)}
                    ))
                current_chunk = sentence + " "
        
        # Add last chunk
        if current_chunk:
            chunks.append(Document(
                page_content=current_chunk.strip(),
                metadata={"source_file": source_filename, "chunk_id": len(chunks)}
            ))
        
        return chunks

    def get_available_hoas(self) -> List[str]:
        hoa_folders = []
        if not self.hoa_base_dir.exists():
            return hoa_folders
            
        for hoa_name in sorted(os.listdir(self.hoa_base_dir)):
            hoa_path = self.hoa_base_dir / hoa_name
            if hoa_path.is_dir():
                has_pdf_files = bool(list(glob.glob(str(hoa_path / "**" / "*.pdf"), recursive=True)))
                if has_pdf_files:
                    hoa_folders.append(hoa_name)
        return hoa_folders

    def load_hoa(self, hoa_name: str) -> dict:
        if hoa_name in self.hoa_documents:
            return {"success": True, "message": f"HOA '{hoa_name}' already loaded."}
        
        hoa_path = self.hoa_base_dir / hoa_name
        if not hoa_path.is_dir():
            return {"success": False, "message": f"HOA folder '{hoa_name}' not found."}

        # Load embedding model
        if self.embedding_model is None:
            self.embedding_model = self.get_embedding_model()

        # Process PDFs
        all_documents = []
        pdf_files = list(glob.glob(str(hoa_path / "**" / "*.pdf"), recursive=True))
        
        progress_bar = st.progress(0)
        for i, pdf_path in enumerate(pdf_files):
            progress_bar.progress((i + 1) / len(pdf_files))
            st.write(f"Processing: {os.path.basename(pdf_path)}")
            
            pdf_content = self.extract_pdf_text(pdf_path)
            if pdf_content:
                chunks = self.create_chunks(pdf_content, os.path.basename(pdf_path))
                all_documents.extend(chunks)

        if not all_documents:
            return {"success": False, "message": f"No readable PDF files found for HOA: {hoa_name}"}

        # Create embeddings
        st.write("Creating embeddings...")
        texts = [doc.page_content for doc in all_documents]
        embeddings = self.embedding_model.encode(texts)
        
        # Store
        self.hoa_documents[hoa_name] = all_documents
        self.hoa_embeddings[hoa_name] = embeddings
        
        return {
            "success": True,
            "message": f"Successfully loaded HOA: {hoa_name}",
            "document_count": len(pdf_files),
            "chunk_count": len(all_documents)
        }

    def query_hoa(self, question: str, hoa_name: str, k: int = 3) -> dict:
        if hoa_name not in self.hoa_documents:
            return {"error": f"HOA '{hoa_name}' not loaded. Please load it first."}

        # Get question embedding
        question_embedding = self.embedding_model.encode([question])
        
        # Calculate similarities
        similarities = cosine_similarity(question_embedding, self.hoa_embeddings[hoa_name])[0]
        
        # Get top k results
        top_indices = np.argsort(similarities)[-k:][::-1]
        retrieved_chunks = [self.hoa_documents[hoa_name][i] for i in top_indices]
        
        # Create context
        context = "\n\n".join([
            f"[Document: {doc.metadata.get('source_file')}]\n{doc.page_content}"
            for doc in retrieved_chunks
        ])
        sources = list(set(doc.metadata.get('source_file', 'Unknown') for doc in retrieved_chunks))

        # Generate answer
        prompt = f"""You are a compliance analyst reviewing HOA governing documents.

QUESTION:
{question}

DOCUMENTS:
{context}

RESPONSE GUIDELINES:
- Use only the provided documents
- Keep responses concise and factual
- Always cite specific section references when available (e.g., "Section 7.4(b)", "Article III", "Section 2.1")
ANSWER:"""

        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Cheaper model
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024
            )
            answer = completion.choices[0].message.content
            return {"answer": answer, "sources": sources}
        except Exception as e:
            return {"error": str(e)}

# Initialize system
@st.cache_resource
def get_hoa_system():
    return SimpleHOASystem()

# Streamlit UI
def main():
    st.set_page_config(page_title="HOA Document Q&A", layout="wide")
    
    st.title("🏠 HOA Document Q&A System")
    st.markdown("Ask questions about HOA governing documents")

    hoa_system = get_hoa_system()
    available_hoas = hoa_system.get_available_hoas()

    if not available_hoas:
        st.error("No HOA documents found in the data directory")
        return

    # Sidebar
    st.sidebar.header("Available HOAs")
    selected_hoa = st.sidebar.selectbox("Select an HOA:", available_hoas)

    if st.sidebar.button("Load HOA Documents"):
        with st.spinner(f"Loading {selected_hoa} documents..."):
            result = hoa_system.load_hoa(selected_hoa)
            if result["success"]:
                st.sidebar.success(result["message"])
            else:
                st.sidebar.error(result["message"])

    # Main interface
    if selected_hoa in hoa_system.hoa_documents:
        st.success(f"✅ {selected_hoa} documents loaded")
        
        question = st.text_input("Ask a question about the HOA documents:")
        
        if st.button("Ask Question") and question:
            with st.spinner("Searching documents..."):
                result = hoa_system.query_hoa(question, selected_hoa)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.markdown("### Answer:")
                    st.write(result["answer"])
                    
                    st.markdown("### Sources:")
                    for source in result["sources"]:
                        st.write(f"• {source}")
    else:
        st.info(f"Please load {selected_hoa} documents first using the sidebar.")

if __name__ == "__main__":
    main()