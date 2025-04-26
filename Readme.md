```bash
# Multimodal Retrieval-Augmented Generation (RAG) System

This project implements a Multimodal Retrieval-Augmented Generation (RAG) system that extracts, summarizes, and retrieves information from complex PDFs containing text, tables, and images. It uses OpenAI’s GPT-3.5-turbo and GPT-4o models, LangChain, ChromaDB, and Unstructured parsing tools.

# 🚀 Project Overview
- PDF Processing: Extracts text, tables, and images from PDFs using Unstructured.
- LLM Summarization:
  - Text and Table summaries generated using GPT-3.5-Turbo.
  - Image understanding using GPT-4o (Vision capabilities).
- Vector Storage: Embeddings stored in ChromaDB for semantic search.
- Document Storage: Raw elements stored in InMemoryStore for complete retrieval.
- Retrieval-Augmented QA: Context-based answering using LangChain’s retrieval system.

# 🛠️ Technologies Used
- Python 3.10+
- LangChain
- OpenAI API (GPT-3.5-Turbo, GPT-4o)
- ChromaDB
- Unstructured
- Pytesseract
- dotenv

# 📦 Installation
git clone https://github.com/yourusername/multimodal-rag-system.git
cd multimodal-rag-system
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt

# Create a .env file inside your project directory with the following content:
# OPENAI_API_KEY=your_openai_api_key_here
# CHROMA_COLLECTION_NAME=multimodal_rag_project
# ENVIRONMENT=development
# DEBUG=True
# LOG_LEVEL=info
# TESSERACT_CMD_PATH=C:/Program Files/Tesseract-OCR/tesseract.exe

# Install Tesseract OCR manually:
# Windows: https://github.com/tesseract-ocr/tesseract
# macOS (Homebrew): brew install tesseract
# Linux: sudo apt install tesseract-ocr

# 📋 How to Run
python multimodal_rag_system.py

# Edit your script if needed to match the PDF filename:
# filename = os.path.join(input_path, "your_pdf_file.pdf")

# Example queries you can ask:
# What products are displayed in the images?
# What is the financial summary of the company?
# What is the total revenue mentioned?

# ✨ Features
# - Full multimodal processing: text, tables, and images.
# - Summarization and extraction using different LLMs.
# - Semantic vector-based search and retrieval.
# - OCR support for non-readable images.
# - Flexible and scalable architecture using ChromaDB and LangChain.

# 📚 Learning Outcomes
# - Building real-world Retrieval-Augmented Generation systems.
# - Summarizing multiple modalities with different LLMs.
# - Building a QA pipeline using LangChain.
# - Secure environment variable handling with dotenv.

# 📜 License
# This project is licensed under the MIT License.

# 🤝 Contributing
# Contributions are welcome! Please fork the repository and submit a pull request.
```
