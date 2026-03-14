# 🔬 sci_synth

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)

**AI-powered academic paper analysis with RAG (Groq + HuggingFace)**

Analyze research papers and scholarly articles with natural language queries. Built with Groq's Llama 3.3 (fast inference) and HuggingFace embeddings (local, unlimited).

## ✨ Features

- 📄 **Smart PDF Processing** - Automatic parsing and intelligent chunking
- 💬 **Q&A System** - Ask questions with source citations
- ✅ **Question Relevance Gate** - Detects off-topic questions before LLM generation
- ⚠️ **Contradiction Detection** - Find conflicting claims
- 🎯 **Gap Analysis** - Identify research opportunities
- 🧭 **Retrieval Insights** - Chunk count, relevance score, and source provenance
- 🎨 **Futuristic Streamlit UI** - Neat, focused dashboard

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
# (API key is read from .env only and is not shown in the dashboard)

# Run
streamlit run src/app.py
```

Open `http://localhost:8501` and start analyzing papers!

## 📖 Usage

### Streamlit Dashboard
Upload PDFs → Ask a question → Relevance check → Answer with sources + retrieval diagnostics

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

## 📁 Current Structure

```
sci_synth/
├── src/
│   ├── ingestion.py        # PDF parsing
│   ├── chunking.py         # Text splitting
│   ├── embeddings.py       # Vector embeddings
│   ├── retrieval.py        # RAG + LLM
│   ├── analysis.py         # Contradictions + gap analysis
│   └── app.py              # Streamlit UI
├── data/                   # Your PDFs
├── .env.example            # API key template
└── requirements.txt
```

## 🆕 What's New (current)

- ✅ Academic-document validation during upload (accept scholarly docs, reject non-academic docs)
- ✅ Hidden API-key handling via `.env` only
- ✅ Relevance gate for user questions before generation
- ✅ Retrieval observability (chunks, relevance, provenance)
- ✅ Streamlined UI (removed synthesis/citation tabs and extra controls)

## 🛠️ Tech Stack

LangChain • Streamlit • Groq • HuggingFace • Chroma • PyMuPDF • NetworkX

## 🐛 Troubleshooting

**Rate limits?** Groq free tier resets every minute  
**Missing modules?** Run `pip install --upgrade -r requirements.txt`  
**No .env file?** Copy `.env.example` to `.env`  
**No answers for a question?** The relevance gate may be filtering off-topic prompts—ask paper-grounded questions (methods/findings/limitations).


## 🙏 Acknowledgments

Built with Groq, HuggingFace, LangChain, Streamlit & Chroma

---


*Built for researcher. 100% free to use.* 🔬✨
