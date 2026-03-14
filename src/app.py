"""
Streamlit Dashboard for sci_synth RAG System
Interactive UI for scientific paper analysis and synthesis
"""

import streamlit as st
import os
import sys
from pathlib import Path
import tempfile
from dotenv import load_dotenv
from typing import Dict, List, Optional

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from ingestion import PaperIngestion
from chunking import TextChunker
from embeddings import EmbeddingManager
from retrieval import RAGRetrieval
from analysis import ContradictionDetector, ResearchGapAnalyzer


# Page configuration
st.set_page_config(
    page_title="sci_synth - Scientific Paper RAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at top right, #1f2a44 0%, #0b1220 45%, #060b16 100%);
    }
    .block-container {
        padding-top: 1.5rem;
    }
    .main-header {
        font-size: 2.8rem;
        color: #8ec5ff;
        font-weight: bold;
        margin-bottom: 0.3rem;
        text-shadow: 0 0 20px rgba(90, 170, 255, 0.45);
    }
    .sub-header {
        font-size: 1.2rem;
        color: #c8d7ef;
        margin-bottom: 1.4rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.8rem;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(146, 181, 255, 0.18);
        border-radius: 16px;
        padding: 0.45rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        border-radius: 10px;
        padding: 0px 16px;
        color: #dce8ff;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #2756a6, #2b7de9);
    }
    .insight-card {
        border: 1px solid rgba(157, 197, 255, 0.22);
        border-radius: 14px;
        background: rgba(13, 21, 38, 0.75);
        padding: 0.95rem 1rem;
        margin: 0.5rem 0;
    }
    .insight-title {
        font-weight: 700;
        color: #9cc5ff;
        margin-bottom: 0.35rem;
    }
    .tiny-muted {
        color: #b8c6dd;
        font-size: 0.9rem;
    }
    .status-badge {
        display: inline-block;
        background: rgba(43, 125, 233, 0.2);
        color: #b5d5ff;
        border: 1px solid rgba(69, 148, 255, 0.35);
        border-radius: 999px;
        padding: 0.2rem 0.6rem;
        font-size: 0.82rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'papers_loaded' not in st.session_state:
        st.session_state.papers_loaded = False
    if 'papers_data' not in st.session_state:
        st.session_state.papers_data = []
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'embedding_manager' not in st.session_state:
        st.session_state.embedding_manager = None
    if 'documents' not in st.session_state:
        st.session_state.documents = []


def format_relevance_label(score: float) -> str:
    """Human-friendly relevance label."""
    if score >= 0.55:
        return "High"
    if score >= 0.35:
        return "Medium"
    return "Low"


def check_question_relevance(
    vectorstore,
    question: str,
    paper_id: Optional[str] = None,
    k: int = 5
) -> Dict:
    """
    Relevance gate before LLM answer generation.
    Uses vector similarity to estimate whether the question maps to indexed paper context.
    """
    filter_dict = {"paper_id": paper_id} if paper_id else None
    if filter_dict:
        scored = vectorstore.similarity_search_with_score(question, k=k, filter=filter_dict)
    else:
        scored = vectorstore.similarity_search_with_score(question, k=k)

    if not scored:
        return {
            "is_relevant": False,
            "avg_relevance": 0.0,
            "max_relevance": 0.0,
            "scores": [],
            "threshold": 0.30,
            "reason": "No matching context chunks were found for the question."
        }

    relevance_scores = [1 / (1 + float(distance)) for _, distance in scored]
    avg_relevance = sum(relevance_scores) / len(relevance_scores)
    max_relevance = max(relevance_scores)

    # Soft gate tuned to avoid unnecessary LLM calls for off-topic questions.
    threshold = 0.30
    is_relevant = max_relevance >= threshold

    return {
        "is_relevant": is_relevant,
        "avg_relevance": avg_relevance,
        "max_relevance": max_relevance,
        "scores": relevance_scores,
        "threshold": threshold,
        "reason": (
            "Question appears relevant to indexed academic content."
            if is_relevant else
            "Question seems weakly related to uploaded documents. Please ask about methods, findings, limitations, or conclusions from the uploaded papers."
        )
    }


def render_pipeline_overview(last_relevance: Optional[Dict] = None):
    """Show processing and retrieval pipeline status."""
    total_chunks = len(st.session_state.chunks) if st.session_state.get("chunks") else 0
    total_docs = len(st.session_state.papers_data) if st.session_state.get("papers_data") else 0
    avg_rel = (last_relevance or {}).get("avg_relevance", 0.0)

    st.markdown(
        """
        <div class="insight-card">
            <div class="insight-title">⚙️ Retrieval Pipeline Status</div>
            <span class="status-badge">Ingestion → Chunking → Embedding → Vector Search → LLM Synthesis</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Indexed Documents", total_docs)
    with col2:
        st.metric("Total Chunks", total_chunks)
    with col3:
        st.metric("Embedding Store", "Chroma")
    with col4:
        st.metric("Last Avg Relevance", f"{avg_rel:.3f}")


def render_retrieval_insights(result: Dict):
    """Render retrieval provenance and relevance details."""
    summary = result.get("retrieval_summary", {})
    diagnostics: List[Dict] = result.get("retrieval_diagnostics", [])

    st.markdown("### 🧭 Retrieval Insights")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mode", summary.get("retrieval_mode", "vector_similarity"))
    with col2:
        st.metric("Chunks Retrieved (k)", summary.get("k", len(diagnostics)))
    with col3:
        st.metric("Avg Relevance", f"{summary.get('avg_relevance', 0.0):.3f}")

    if diagnostics:
        with st.expander("🔎 View chunk-level relevance and source provenance", expanded=False):
            for i, item in enumerate(diagnostics, 1):
                rel = item.get("relevance_score", 0.0)
                rel_label = format_relevance_label(rel)
                st.markdown(
                    f"**{i}. {item.get('paper_title', 'Unknown')}** | "
                    f"section: `{item.get('section', 'Unknown')}` | "
                    f"chunk: `{item.get('chunk_index', 'N/A')}` | "
                    f"relevance: **{rel:.3f} ({rel_label})**"
                )
                st.caption(item.get("preview", ""))
                st.markdown("---")


def render_non_relevant_question_message(relevance_check: Dict, scope_label: str):
    """Render a user-friendly message when question relevance is too low."""
    st.warning(
        f"I couldn’t find strong evidence that this question maps to the {scope_label} context."
    )
    st.info(relevance_check.get("reason", "Try rephrasing and grounding your question in uploaded paper content."))
    st.caption(
        f"Max relevance: {relevance_check.get('max_relevance', 0.0):.3f} | "
        f"Threshold: {relevance_check.get('threshold', 0.30):.3f}"
    )
    st.markdown("**Try asking like this:**")
    st.markdown(
        "- What methodology does the paper use?\n"
        "- What are the key findings and limitations?\n"
        "- Which dataset/experimental setup is used?\n"
        "- What future work is suggested?"
    )


def sidebar():
    """Render sidebar with configuration and data loading"""
    st.sidebar.markdown("# 🔬 sci_synth")
    st.sidebar.markdown("### Configuration")

    # API key is read from .env only (not shown in UI)
    has_groq_key = bool(os.getenv("GROQ_API_KEY"))
    if has_groq_key:
        st.sidebar.success("✓ API key loaded from .env")
    else:
        st.sidebar.warning("No GROQ_API_KEY found in .env")
        st.sidebar.caption("Add GROQ_API_KEY in .env to enable question answering.")
    
    # Data Loading
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📄 Load Papers")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF Papers",
        type=['pdf'],
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    
    if st.sidebar.button("🚀 Process Papers", type="primary"):
        if not uploaded_files:
            st.sidebar.error("Please upload PDF files first!")
        elif not has_groq_key:
            st.sidebar.error("Please add GROQ_API_KEY in .env first.")
        else:
            process_papers(uploaded_files)
    
    # Status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Status")
    if st.session_state.papers_loaded:
        st.sidebar.success(f"✓ {len(st.session_state.papers_data)} papers loaded")
        st.sidebar.info(f"✓ {len(st.session_state.chunks)} chunks indexed")
    else:
        st.sidebar.warning("No papers loaded yet")
    
    # Example Data
    if not st.session_state.papers_loaded:
        if st.sidebar.button("📚 Load Example Data"):
            load_example_data("Groq")


def process_papers(uploaded_files):
    """Process uploaded PDF papers"""
    with st.spinner("Processing papers..."):
        try:
            # Step 1: Ingestion
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Step 1/4: Extracting text from PDFs...")
            ingestion = PaperIngestion()
            papers_data = []
            rejected_docs = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                # Process paper
                paper = ingestion.process_paper(tmp_path, paper_id=f"paper_{i+1}")

                # Validate academic/research-oriented structure
                validation = ingestion.assess_academic_document(paper)
                if validation["is_academic_document"]:
                    papers_data.append(paper)
                else:
                    rejected_docs.append({
                        "filename": uploaded_file.name,
                        "reason": validation["reasons"][0] if validation["reasons"] else "Document does not look like an academic/research article.",
                        "score": validation["score"]
                    })
                
                # Cleanup
                os.unlink(tmp_path)
                
                progress_bar.progress((i + 1) / len(uploaded_files) * 0.25)

            if not papers_data:
                raise ValueError(
                    "None of the uploaded files passed academic-document validation. "
                    "Please upload research papers or scholarly articles (with sections like abstract/introduction/references)."
                )
            
            # Step 2: Chunking
            status_text.text("Step 2/4: Chunking text...")
            chunker = TextChunker(chunk_size=400, chunk_overlap=100)
            chunks = chunker.chunk_multiple_papers(papers_data)
            progress_bar.progress(0.5)
            
            # Step 3: Embeddings
            status_text.text("Step 3/4: Creating embeddings...")
            # Use HuggingFace for embeddings (free, local, no API limits)
            embedding_manager = EmbeddingManager(
                provider="huggingface",
                persist_directory="./chroma_db"
            )
            vectorstore = embedding_manager.create_vectorstore(chunks)
            documents = embedding_manager.chunks_to_documents(chunks)
            progress_bar.progress(0.75)
            
            # Step 4: Save to session
            status_text.text("Step 4/4: Finalizing...")
            st.session_state.papers_data = papers_data
            st.session_state.chunks = chunks
            st.session_state.vectorstore = vectorstore
            st.session_state.embedding_manager = embedding_manager
            st.session_state.documents = documents
            st.session_state.papers_loaded = True
            
            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")
            
            st.success(f"Successfully processed {len(papers_data)} academic/research document(s)!")

            if rejected_docs:
                st.warning(f"Rejected {len(rejected_docs)} non-academic document(s).")
                with st.expander("See rejected files and reasons"):
                    for rejected in rejected_docs:
                        st.markdown(
                            f"- **{rejected['filename']}** (score: {rejected['score']}/11): {rejected['reason']}"
                        )

            st.balloons()
            
        except Exception as e:
            st.error(f"Error processing papers: {str(e)}")


def load_example_data(api_provider):
    """Load example/demo data"""
    with st.spinner("Loading example data..."):
        st.info("In production, this would load pre-processed example papers. "
                "For now, please upload your own PDF files.")


def tab_qa():
    """Q&A Tab - Single paper queries"""
    st.markdown("## 💬 Question & Answer")
    st.markdown("Ask questions about specific papers in your collection.")
    
    if not st.session_state.papers_loaded:
        st.warning("⚠️ Please load papers first using the sidebar.")
        return

    last_relevance = st.session_state.get("last_question_relevance")
    render_pipeline_overview(last_relevance=last_relevance)
    
    # Paper selection
    paper_options = {
        p['metadata']['title']: p['paper_id'] 
        for p in st.session_state.papers_data
    }
    
    selected_paper_title = st.selectbox(
        "Select Paper",
        options=list(paper_options.keys())
    )
    selected_paper_id = paper_options[selected_paper_title]
    
    # Question input
    question = st.text_input(
        "Ask a question about this paper:",
        placeholder="What are the main findings of this research?"
    )
    
    # Fixed retrieval size for cleaner UX
    k_results = 5
    
    if st.button("🔍 Ask", type="primary"):
        if not question:
            st.warning("Please enter a question to continue.")
            return
        
        with st.spinner("Searching and generating answer..."):
            try:
                # Step A: relevance gate
                relevance_check = check_question_relevance(
                    vectorstore=st.session_state.vectorstore,
                    question=question,
                    paper_id=selected_paper_id,
                    k=5
                )
                st.session_state.last_question_relevance = relevance_check

                if not relevance_check["is_relevant"]:
                    render_non_relevant_question_message(
                        relevance_check=relevance_check,
                        scope_label="selected paper"
                    )
                    return

                # Use Groq for LLM (fast & free)
                rag = RAGRetrieval(
                    vectorstore=st.session_state.vectorstore,
                    llm_provider="groq"
                )
                
                result = rag.query_single_paper(
                    question=question,
                    paper_id=selected_paper_id,
                    k=k_results
                )
                
                # Display answer
                st.markdown("### 📝 Answer")
                st.markdown(result['answer'])

                render_retrieval_insights(result)
                
                # Display sources
                st.markdown("### 📚 Sources")
                for i, doc in enumerate(result['source_documents'], 1):
                    with st.expander(f"Source {i} - [{doc.metadata.get('section', 'Unknown')}]"):
                        st.text(doc.page_content)
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")


def tab_contradictions():
    """Contradictions Detection Tab"""
    st.markdown("## ⚠️ Contradiction Detection")
    st.markdown("Identify contradictory findings across papers.")
    
    if not st.session_state.papers_loaded:
        st.warning("⚠️ Please load papers first using the sidebar.")
        return
    
    topic = st.text_input(
        "Research Topic",
        placeholder="machine learning performance"
    )

    # Fixed analysis size for simpler UX
    k_chunks = 10
    
    if st.button("🔍 Detect Contradictions", type="primary"):
        if not topic:
            st.error("Please enter a research topic.")
            return
        
        with st.spinner("Analyzing papers for contradictions..."):
            try:
                # Use Groq for LLM (fast & free)
                detector = ContradictionDetector(
                    vectorstore=st.session_state.vectorstore,
                    llm_provider="groq"
                )
                
                contradictions = detector.detect_contradictions(
                    topic=topic,
                    k=k_chunks
                )
                
                if contradictions:
                    st.success(f"Found {len(contradictions)} potential contradiction(s)")
                    
                    for i, contradiction in enumerate(contradictions, 1):
                        with st.expander(f"🔴 Contradiction {i}", expanded=True):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Paper 1:**")
                                st.info(contradiction['paper1_title'])
                            
                            with col2:
                                st.markdown("**Paper 2:**")
                                st.info(contradiction['paper2_title'])
                            
                            st.markdown("**Analysis:**")
                            st.markdown(contradiction['explanation'])
                else:
                    st.info("No contradictions detected in the analyzed papers.")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")


def tab_gaps():
    """Research Gaps Tab"""
    st.markdown("## 🎯 Research Gaps")
    st.markdown("Identify research gaps and future work opportunities.")
    
    if not st.session_state.papers_loaded:
        st.warning("⚠️ Please load papers first using the sidebar.")
        return

    # Fixed analysis size for simpler UX
    k_docs = 20
    
    if st.button("🔍 Find Research Gaps", type="primary"):
        with st.spinner("Analyzing papers for research gaps..."):
            try:
                # Use Groq for LLM (fast & free)
                analyzer = ResearchGapAnalyzer(
                    vectorstore=st.session_state.vectorstore,
                    llm_provider="groq"
                )
                
                gaps = analyzer.find_research_gaps(k=k_docs)
                
                if gaps:
                    st.success(f"Identified {len(gaps)} research gap(s)")
                    
                    for i, gap_entry in enumerate(gaps, 1):
                        with st.expander(f"💡 Gap {i} - {gap_entry['paper_title'][:50]}...", expanded=i<=3):
                            st.markdown(f"**From:** {gap_entry['paper_title']}")
                            st.markdown(f"**Gap:** {gap_entry['gap']}")
                else:
                    st.info("No specific research gaps identified.")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")


def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🔬 sci_synth</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Scientific Paper RAG & Analysis System</div>', unsafe_allow_html=True)
    
    # Sidebar
    sidebar()
    
    # Main tabs
    tabs = st.tabs([
        "💬 Q&A",
        "⚠️ Contradictions",
        "🎯 Research Gaps"
    ])
    
    with tabs[0]:
        tab_qa()
    
    with tabs[1]:
        tab_contradictions()
    
    with tabs[2]:
        tab_gaps()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with LangChain, Chroma, and Streamlit | "
        "<a href='https://github.com/yourusername/sci_synth'>GitHub</a>"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
