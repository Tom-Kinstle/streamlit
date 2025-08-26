# HOA Document Q&A System - Render Deployment

A Streamlit application that allows users to ask questions about HOA governing documents using AI and RAG (Retrieval-Augmented Generation).

## Features

- 🏠 **Multiple HOA Support**: Select from different HOA document collections
- 🤖 **AI-Powered Q&A**: Uses OpenAI GPT-4o-mini for intelligent responses
- 📚 **Document Retrieval**: Advanced semantic search through HOA documents
- 💬 **Interactive UI**: Clean Streamlit interface with example questions
- 🔍 **Source Citations**: Shows which documents answers came from

## Quick Deploy to Render

1. **Fork this repository** to your GitHub account

2. **Connect to Render:**
   - Go to [render.com](https://render.com) and sign up/login
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

3. **Configure Environment:**
   - Set `OPENAI_API_KEY` in Render dashboard
   - Render will automatically use `render.yaml` configuration

4. **Deploy:**
   - Click "Create Web Service"
   - Render will build and deploy automatically

## Local Development

1. **Clone and setup:**
   ```bash
   git clone <your-repo>
   cd hoa-qa-system
   pip install -r requirements.txt
   ```

2. **Environment setup:**
   ```bash
   cp .env.template .env
   # Edit .env with your OpenAI API key
   ```

3. **Add HOA documents:**
   ```
   data/
   └── hoa_documents/
       ├── HOA_Name_1/
       │   ├── bylaws.pdf
       │   └── covenants.pdf
       └── HOA_Name_2/
           └── rules.pdf
   ```

4. **Run locally:**
   ```bash
   streamlit run app.py
   ```

## File Structure

```
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── render.yaml        # Render deployment configuration
├── logo.jpg           # Application logo
├── .env.template      # Environment variables template
└── data/              # HOA documents directory
    └── hoa_documents/
        └── [HOA folders with PDFs]
```

## Environment Variables

- `OPENAI_API_KEY` (required): Your OpenAI API key
- `HOA_BASE_DIR` (optional): Custom path to HOA documents

## Architecture

- **Frontend**: Streamlit web interface
- **Backend**: Python with LangChain for document processing
- **AI Model**: OpenAI GPT-4o-mini
- **Embeddings**: HuggingFace BGE-large-en-v1.5
- **Vector Store**: ChromaDB with persistent storage
- **PDF Processing**: PyMuPDF for text extraction

## Usage

1. Select an HOA from the dropdown
2. Wait for documents to load (first time only)
3. Ask questions or click example questions
4. View AI-generated answers with source citations

## Example Questions

- "Number of Directors?"
- "Term of Office?"
- "Candidate Qualifications/Eligibility Requirements?"
- "Are write-ins allowed?"
- "Is there a process for handling tie votes?"

## Deployment Notes

- Uses Render's free tier (512MB RAM)
- Persistent disk for vector store caching
- Automatic HTTPS and custom domains available
- Health checks configured for reliability

## Support

For issues or questions, please check the GitHub repository issues section.