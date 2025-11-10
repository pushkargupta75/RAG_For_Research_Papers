"""
Streamlit Dashboard for sci_synth RAG System
Interactive UI for scientific paper analysis and synthesis
"""

import streamlit as st
import os
import sys
from pathlib import Path
import json
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from ingestion import PaperIngestion
from chunking import TextChunker
from embeddings import EmbeddingManager
from retrieval import RAGRetrieval
from analysis import ContradictionDetector, ResearchGapAnalyzer, CitationNetworkBuilder


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
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 20px;
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


def sidebar():
    """Render sidebar with configuration and data loading"""
    st.sidebar.markdown("# 🔬 sci_synth")
    st.sidebar.markdown("### Configuration")
    
    # API Key Configuration
    with st.sidebar.expander("🔑 API Keys", expanded=False):
        api_provider = st.selectbox(
            "LLM Provider",
            ["Groq (Recommended - Fast & Free)", "Google Gemini"],
            key="api_provider"
        )
        
        if "Groq" in api_provider:
            api_key = st.text_input(
                "Groq API Key",
                type="password",
                value=os.getenv("GROQ_API_KEY", ""),
                help="Get free key at: https://console.groq.com",
                key="groq_key"
            )
            if api_key:
                os.environ["GROQ_API_KEY"] = api_key
        else:
            api_key = st.text_input(
                "Google API Key",
                type="password",
                value=os.getenv("GOOGLE_API_KEY", ""),
                help="Get key at: https://makersuite.google.com/app/apikey",
                key="google_key"
            )
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
    
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
        elif not api_key:
            st.sidebar.error("Please provide an API key first!")
        else:
            process_papers(uploaded_files, api_provider)
    
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
            load_example_data(api_provider if 'api_provider' in locals() else "OpenAI")


def process_papers(uploaded_files, api_provider):
    """Process uploaded PDF papers"""
    with st.spinner("Processing papers..."):
        try:
            # Step 1: Ingestion
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Step 1/4: Extracting text from PDFs...")
            ingestion = PaperIngestion()
            papers_data = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                # Process paper
                paper = ingestion.process_paper(tmp_path, paper_id=f"paper_{i+1}")
                papers_data.append(paper)
                
                # Cleanup
                os.unlink(tmp_path)
                
                progress_bar.progress((i + 1) / len(uploaded_files) * 0.25)
            
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
            
            st.success(f"Successfully processed {len(papers_data)} papers!")
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
    
    col1, col2 = st.columns([1, 4])
    with col1:
        k_results = st.number_input("Number of chunks", min_value=1, max_value=20, value=5)
    
    if st.button("🔍 Ask", type="primary"):
        if not question:
            st.error("Please enter a question.")
            return
        
        with st.spinner("Searching and generating answer..."):
            try:
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
                
                # Display sources
                st.markdown("### 📚 Sources")
                for i, doc in enumerate(result['source_documents'], 1):
                    with st.expander(f"Source {i} - [{doc.metadata.get('section', 'Unknown')}]"):
                        st.text(doc.page_content)
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")


def tab_synthesis():
    """Multi-Paper Synthesis Tab"""
    st.markdown("## 🔬 Multi-Paper Synthesis")
    st.markdown("Synthesize information across multiple papers.")
    
    if not st.session_state.papers_loaded:
        st.warning("⚠️ Please load papers first using the sidebar.")
        return
    
    # Question input
    question = st.text_area(
        "Research Question",
        placeholder="What are the common methodologies used across these papers?",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        k_results = st.number_input("Chunks to retrieve", min_value=5, max_value=50, value=10)
    
    if st.button("🔍 Synthesize", type="primary"):
        if not question:
            st.error("Please enter a research question.")
            return
        
        with st.spinner("Analyzing multiple papers..."):
            try:
                # Use Groq for LLM (fast & free)
                rag = RAGRetrieval(
                    vectorstore=st.session_state.vectorstore,
                    llm_provider="groq"
                )
                
                result = rag.query_multi_paper(
                    question=question,
                    k=k_results
                )
                
                # Display answer
                st.markdown("### 📝 Synthesized Answer")
                st.markdown(result['answer'])
                
                # Papers cited
                st.markdown("### 📖 Papers Referenced")
                for paper_id, title in result['papers_cited'].items():
                    st.markdown(f"- **{title}**")
                
                # Source chunks
                with st.expander("📚 View All Source Chunks"):
                    for i, doc in enumerate(result['source_documents'], 1):
                        st.markdown(f"**Source {i}** - {doc.metadata.get('paper_title', 'Unknown')[:50]}...")
                        st.text(doc.page_content[:300] + "...")
                        st.markdown("---")
                        
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
    
    col1, col2 = st.columns([1, 4])
    with col1:
        k_chunks = st.number_input("Chunks to analyze", min_value=5, max_value=30, value=10)
    
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
    
    col1, col2 = st.columns([1, 4])
    with col1:
        k_docs = st.number_input("Documents to analyze", min_value=10, max_value=50, value=20)
    
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


def tab_citations():
    """Citation Network Tab"""
    st.markdown("## 🕸️ Citation Network")
    st.markdown("Visualize the citation relationships between papers.")
    
    if not st.session_state.papers_loaded:
        st.warning("⚠️ Please load papers first using the sidebar.")
        return
    
    if st.button("🔍 Build Citation Network", type="primary"):
        with st.spinner("Building citation network..."):
            try:
                builder = CitationNetworkBuilder()
                builder.build_network(st.session_state.papers_data)
                builder.visualize_network("citation_graph.html")
                
                # Display statistics
                stats = builder.get_statistics()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Papers", stats['total_papers'])
                with col2:
                    st.metric("Total Citations", stats['total_citations'])
                with col3:
                    st.metric("Avg Citations/Paper", f"{stats['avg_citations_per_paper']:.2f}")
                
                # Display network
                st.markdown("### 📊 Interactive Citation Network")
                
                if Path("citation_graph.html").exists():
                    with open("citation_graph.html", "r", encoding="utf-8") as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=800, scrolling=True)
                else:
                    st.warning("Citation graph not generated.")
                    
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
        "🔬 Multi-Paper Synthesis",
        "⚠️ Contradictions",
        "🎯 Research Gaps",
        "🕸️ Citation Network"
    ])
    
    with tabs[0]:
        tab_qa()
    
    with tabs[1]:
        tab_synthesis()
    
    with tabs[2]:
        tab_contradictions()
    
    with tabs[3]:
        tab_gaps()
    
    with tabs[4]:
        tab_citations()
    
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
