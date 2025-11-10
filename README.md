# 🔬 sci_synth

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)

**AI-powered scientific paper analysis with RAG - 100% free using Groq & HuggingFace**

Analyze research papers with natural language queries. Built with Groq's Llama 3.3 (ultra-fast LLM) and HuggingFace embeddings (local, unlimited).

## ✨ Features

- 📄 **Smart PDF Processing** - Automatic parsing and intelligent chunking
- 💬 **Q&A System** - Ask questions with source citations
- 🔬 **Multi-Paper Synthesis** - Compare insights across documents
- ⚠️ **Contradiction Detection** - Find conflicting claims
- 🎯 **Gap Analysis** - Identify research opportunities
- 🕸️ **Citation Networks** - Interactive paper relationships
- 🎨 **Streamlit UI** - Clean web interface

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/sci_synth.git
cd sci_synth
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Get free API key from https://console.groq.com
# Copy .env.example to .env and add: GROQ_API_KEY=your_key

# Run
streamlit run src/app.py
```

Open `http://localhost:8501` and start analyzing papers!

## 📖 Usage

### Streamlit Dashboard
Upload PDFs → Ask questions → Get answers with citations

### Python API
```python
from src.ingestion import PaperIngestion
from src.retrieval import RAGRetrieval

# Process paper
paper = PaperIngestion().process_paper("paper.pdf")

# Query with RAG
rag = RAGRetrieval(vectorstore, llm_provider="groq")
result = rag.query_single_paper("What are the main findings?", "001")
```

## 💡 Why This Stack?

| Component | Choice | Why |
|-----------|--------|-----|
| **LLM** | Groq (Llama 3.3) | 500+ tok/sec, free tier |
| **Embeddings** | HuggingFace | 100% local, no limits |
| **Vector DB** | Chroma | Fast, persistent storage |
| **Framework** | LangChain | Easy RAG orchestration |

**Result:** Fast, free, and privacy-friendly (your papers never leave your machine!)

## 📁 Structure

```
sci_synth/
├── src/
│   ├── ingestion.py        # PDF parsing
│   ├── chunking.py         # Text splitting
│   ├── embeddings.py       # Vector embeddings
│   ├── retrieval.py        # RAG + LLM
│   ├── analysis.py         # Advanced features
│   └── app.py              # Streamlit UI
├── notebooks/              # 5 tutorials
├── data/                   # Your PDFs
└── requirements.txt
```

## 🆕 What's New (v2.0)

- ✅ Upgraded to Llama 3.3 70B (faster + smarter)
- ✅ Switched to HuggingFace embeddings (no API limits!)
- ✅ Updated LangChain API (latest imports)
- ✅ Added GitHub workflows & templates
- ✅ Enhanced security & privacy

## 🛠️ Tech Stack

LangChain • Streamlit • Groq • HuggingFace • Chroma • PyMuPDF • NetworkX

## 🐛 Troubleshooting

**Rate limits?** Groq free tier resets every minute  
**Missing modules?** Run `pip install --upgrade -r requirements.txt`  
**No .env file?** Copy `.env.example` to `.env`


## 🙏 Acknowledgments

Built with Groq, HuggingFace, LangChain, Streamlit & Chroma

---


*Built for researcher. 100% free to use.* 🔬✨