"""
Analysis Module: Contradiction Detection, Research Gaps, and Citation Networks
Advanced analysis capabilities for scientific literature
"""

from typing import List, Dict, Optional, Set, Tuple
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import networkx as nx
from pyvis.network import Network
import re
from collections import defaultdict
import os


class ContradictionDetector:
    """
    Detects contradictions between different papers on the same topic
    """
    
    def __init__(
        self,
        vectorstore,
        llm_provider: str = "groq",
        model_name: Optional[str] = None
    ):
        """
        Initialize contradiction detector
        
        Args:
            vectorstore: Vector database instance
            llm_provider: "groq" or "google"
            model_name: Specific model name
        """
        self.vectorstore = vectorstore
        
        # Initialize LLM
        if llm_provider == "groq":
            model_name = model_name or "llama-3.3-70b-versatile"
            self.llm = ChatGroq(
                model=model_name,
                temperature=0.0,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
        elif llm_provider == "google":
            model_name = model_name or "gemini-pro"
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.0,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
    
    def detect_contradictions(
        self,
        topic: str,
        k: int = 10
    ) -> List[Dict]:
        """
        Detect contradictions across papers on a given topic
        
        Args:
            topic: Research topic to analyze
            k: Number of chunks to retrieve
            
        Returns:
            List of detected contradictions
        """
        # Retrieve relevant chunks
        docs = self.vectorstore.similarity_search(topic, k=k)
        
        # Group by paper
        papers_content = defaultdict(list)
        for doc in docs:
            paper_id = doc.metadata.get("paper_id", "Unknown")
            papers_content[paper_id].append({
                "text": doc.page_content,
                "section": doc.metadata.get("section", "Unknown"),
                "title": doc.metadata.get("paper_title", "Unknown")
            })
        
        # Get paper summaries for the topic
        paper_summaries = {}
        for paper_id, chunks in papers_content.items():
            context = "\n".join([c["text"] for c in chunks])
            summary = self._summarize_paper_stance(topic, context, chunks[0]["title"])
            paper_summaries[paper_id] = {
                "title": chunks[0]["title"],
                "summary": summary
            }
        
        # Compare pairs of papers
        contradictions = []
        paper_ids = list(paper_summaries.keys())
        
        for i in range(len(paper_ids)):
            for j in range(i + 1, len(paper_ids)):
                paper1_id = paper_ids[i]
                paper2_id = paper_ids[j]
                
                contradiction = self._compare_papers(
                    topic,
                    paper_summaries[paper1_id],
                    paper_summaries[paper2_id],
                    paper1_id,
                    paper2_id
                )
                
                if contradiction:
                    contradictions.append(contradiction)
        
        return contradictions
    
    def _summarize_paper_stance(
        self,
        topic: str,
        context: str,
        paper_title: str
    ) -> str:
        """
        Summarize a paper's stance on a topic
        """
        template = """Based on the following excerpts from a scientific paper titled "{title}", 
summarize the paper's main findings and stance regarding: {topic}

Excerpts:
{context}

Summary (2-3 sentences, focus on key claims and findings):"""
        
        prompt = template.format(
            title=paper_title,
            topic=topic,
            context=context[:3000]  # Limit context length
        )
        
        return self.llm.invoke(prompt).content
    
    def _compare_papers(
        self,
        topic: str,
        paper1: Dict,
        paper2: Dict,
        paper1_id: str,
        paper2_id: str
    ) -> Optional[Dict]:
        """
        Compare two papers for contradictions
        """
        template = """Compare the following two scientific papers regarding the topic: {topic}

Paper 1: {title1}
{summary1}

Paper 2: {title2}
{summary2}

Determine if these papers contradict each other. Respond with one of:
- "CONTRADICT": The papers make conflicting claims
- "AGREE": The papers support each other's findings
- "UNRELATED": The papers discuss different aspects

Followed by a brief explanation.

Analysis:"""
        
        prompt = template.format(
            topic=topic,
            title1=paper1["title"],
            summary1=paper1["summary"],
            title2=paper2["title"],
            summary2=paper2["summary"]
        )
        
        response = self.llm.invoke(prompt).content
        
        # Parse response
        if "CONTRADICT" in response.upper():
            return {
                "paper1_id": paper1_id,
                "paper1_title": paper1["title"],
                "paper2_id": paper2_id,
                "paper2_title": paper2["title"],
                "topic": topic,
                "explanation": response
            }
        
        return None


class ResearchGapAnalyzer:
    """
    Identifies research gaps and future work opportunities
    """
    
    def __init__(
        self,
        vectorstore,
        llm_provider: str = "groq",
        model_name: Optional[str] = None
    ):
        """
        Initialize research gap analyzer
        
        Args:
            vectorstore: Vector database instance
            llm_provider: "groq" or "google"
            model_name: Specific model name
        """
        self.vectorstore = vectorstore
        
        if llm_provider == "groq":
            model_name = model_name or "llama-3.3-70b-versatile"
            self.llm = ChatGroq(
                model=model_name,
                temperature=0.0,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
        elif llm_provider == "google":
            model_name = model_name or "gemini-pro"
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.0,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
    
    def find_research_gaps(
        self,
        domain: Optional[str] = None,
        k: int = 20
    ) -> List[Dict]:
        """
        Identify research gaps from papers
        
        Args:
            domain: Optional specific domain to focus on
            k: Number of documents to analyze
            
        Returns:
            List of identified research gaps
        """
        # Search for limitation and future work sections
        queries = [
            "limitations of this study",
            "future work",
            "future research",
            "open questions",
            "challenges remain"
        ]
        
        all_docs = []
        for query in queries:
            docs = self.vectorstore.similarity_search(query, k=k//len(queries))
            all_docs.extend(docs)
        
        # Filter for relevant sections
        gap_docs = []
        for doc in all_docs:
            section = doc.metadata.get("section", "").lower()
            if any(keyword in section for keyword in ["limitation", "conclusion", "discussion", "future"]):
                gap_docs.append(doc)
        
        # Extract gaps from documents
        gaps = []
        for doc in gap_docs:
            extracted_gaps = self._extract_gaps(doc)
            if extracted_gaps:
                gaps.append({
                    "paper_id": doc.metadata.get("paper_id", "Unknown"),
                    "paper_title": doc.metadata.get("paper_title", "Unknown"),
                    "section": doc.metadata.get("section", "Unknown"),
                    "gaps": extracted_gaps,
                    "context": doc.page_content
                })
        
        # Cluster similar gaps
        unique_gaps = self._cluster_gaps(gaps)
        
        return unique_gaps
    
    def _extract_gaps(self, doc: Document) -> List[str]:
        """
        Extract research gap statements from a document
        """
        template = """Extract research gaps, limitations, and future work suggestions from the following text.
List only the specific gaps or future research directions mentioned.

Text:
{text}

Research Gaps (list one per line, or "NONE" if no gaps mentioned):"""
        
        prompt = template.format(text=doc.page_content)
        response = self.llm.invoke(prompt).content
        
        # Parse response
        if "NONE" in response.upper():
            return []
        
        gaps = [line.strip() for line in response.split('\n') if line.strip() and not line.strip().startswith('-')]
        return [gap.lstrip('•-*123456789. ') for gap in gaps if len(gap) > 20]
    
    def _cluster_gaps(self, gaps: List[Dict]) -> List[Dict]:
        """
        Cluster similar gaps together
        """
        # Simple clustering based on semantic similarity
        # In production, could use more sophisticated clustering
        
        unique_gaps = []
        seen_descriptions = set()
        
        for gap_entry in gaps:
            for gap in gap_entry["gaps"]:
                # Simple deduplication
                gap_lower = gap.lower()
                if not any(gap_lower in seen or seen in gap_lower for seen in seen_descriptions):
                    unique_gaps.append({
                        "gap": gap,
                        "paper_title": gap_entry["paper_title"],
                        "paper_id": gap_entry["paper_id"]
                    })
                    seen_descriptions.add(gap_lower)
        
        return unique_gaps


class CitationNetworkBuilder:
    """
    Builds and visualizes citation networks from papers
    """
    
    def __init__(self):
        """Initialize citation network builder"""
        self.graph = nx.DiGraph()
    
    def extract_citations(self, paper_data: Dict) -> List[str]:
        """
        Extract citations from paper's references section
        
        Args:
            paper_data: Paper dictionary with sections
            
        Returns:
            List of cited paper identifiers
        """
        references_text = ""
        
        # Find references section
        for section_name, section_text in paper_data.get("sections", {}).items():
            if any(keyword in section_name.lower() for keyword in ["reference", "bibliography", "cited"]):
                references_text = section_text
                break
        
        if not references_text:
            return []
        
        # Extract citation patterns (simple heuristic)
        # Looks for patterns like: Author et al. (Year)
        citations = []
        patterns = [
            r'([A-Z][a-z]+(?:\s+et\s+al\.)?)\s*\((\d{4})\)',
            r'([A-Z][a-z]+)\s+and\s+([A-Z][a-z]+)\s*\((\d{4})\)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, references_text)
            citations.extend([str(match) for match in matches])
        
        return citations[:50]  # Limit to avoid clutter
    
    def build_network(self, papers_data: List[Dict]) -> nx.DiGraph:
        """
        Build citation network from papers
        
        Args:
            papers_data: List of paper dictionaries
            
        Returns:
            NetworkX directed graph
        """
        self.graph.clear()
        
        # Add nodes
        for paper in papers_data:
            paper_id = paper["paper_id"]
            title = paper["metadata"]["title"]
            self.graph.add_node(
                paper_id,
                title=title,
                author=paper["metadata"].get("author", "Unknown"),
                year=paper["metadata"].get("year", "Unknown")
            )
        
        # Add edges (citations)
        for paper in papers_data:
            paper_id = paper["paper_id"]
            citations = self.extract_citations(paper)
            
            # Try to match citations to papers in dataset
            for cited in citations:
                for other_paper in papers_data:
                    if paper_id == other_paper["paper_id"]:
                        continue
                    
                    other_title = other_paper["metadata"]["title"]
                    other_author = other_paper["metadata"].get("author", "")
                    
                    # Simple matching heuristic
                    if (str(cited) in other_title or 
                        str(cited) in other_author or
                        other_author in str(cited)):
                        self.graph.add_edge(
                            paper_id,
                            other_paper["paper_id"]
                        )
        
        return self.graph
    
    def visualize_network(
        self,
        output_path: str = "citation_graph.html",
        height: str = "750px",
        width: str = "100%"
    ):
        """
        Create interactive visualization of citation network
        
        Args:
            output_path: Path to save HTML file
            height: Height of visualization
            width: Width of visualization
        """
        net = Network(height=height, width=width, directed=True)
        
        # Add nodes with metadata
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            title = node_data.get("title", node)[:60]
            year = node_data.get("year", "Unknown")
            
            net.add_node(
                node,
                label=f"{title}...\n({year})",
                title=f"{node_data.get('title', node)}\n"
                      f"Author: {node_data.get('author', 'Unknown')}\n"
                      f"Year: {year}",
                color="#97C2FC"
            )
        
        # Add edges
        for edge in self.graph.edges():
            net.add_edge(edge[0], edge[1])
        
        # Configure physics
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -2000,
              "centralGravity": 0.3,
              "springLength": 95
            }
          }
        }
        """)
        
        # Save
        net.save_graph(output_path)
        print(f"✓ Citation network saved to {output_path}")
    
    def get_statistics(self) -> Dict:
        """
        Get network statistics
        
        Returns:
            Dictionary of network metrics
        """
        return {
            "total_papers": self.graph.number_of_nodes(),
            "total_citations": self.graph.number_of_edges(),
            "avg_citations_per_paper": (
                self.graph.number_of_edges() / self.graph.number_of_nodes()
                if self.graph.number_of_nodes() > 0 else 0
            ),
            "most_cited": max(
                self.graph.nodes(),
                key=lambda n: self.graph.in_degree(n)
            ) if self.graph.number_of_nodes() > 0 else None
        }


def main():
    """Example usage"""
    from embeddings import EmbeddingManager
    import json
    
    # Load data
    with open('data/processed_papers.json', 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    # Initialize embeddings
    embedding_manager = EmbeddingManager(provider="google")
    embedding_manager.load_vectorstore()
    
    # Example 1: Detect contradictions
    print("\n=== Detecting Contradictions ===")
    detector = ContradictionDetector(
        vectorstore=embedding_manager.vectorstore,
        llm_provider="groq"
    )
    contradictions = detector.detect_contradictions(
        topic="machine learning performance",
        k=10
    )
    for c in contradictions:
        print(f"Contradiction found between:")
        print(f"  - {c['paper1_title']}")
        print(f"  - {c['paper2_title']}")
    
    # Example 2: Find research gaps
    print("\n=== Finding Research Gaps ===")
    gap_analyzer = ResearchGapAnalyzer(
        vectorstore=embedding_manager.vectorstore,
        llm_provider="groq"
    )
    gaps = gap_analyzer.find_research_gaps(k=20)
    print(f"Found {len(gaps)} unique research gaps")
    for gap in gaps[:5]:
        print(f"  - {gap['gap'][:100]}...")
    
    # Example 3: Build citation network
    print("\n=== Building Citation Network ===")
    builder = CitationNetworkBuilder()
    builder.build_network(papers)
    builder.visualize_network("citation_graph.html")
    stats = builder.get_statistics()
    print(f"Network statistics: {stats}")


if __name__ == "__main__":
    main()
