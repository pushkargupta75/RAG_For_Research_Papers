"""
Retrieval Module with Hybrid Search and LangChain Integration
Combines semantic and keyword-based retrieval for optimal results
"""

from typing import List, Dict, Optional, Tuple
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import numpy as np
import os


class HybridRetriever:
    """
    Combines semantic (vector) and keyword (BM25) retrieval
    """
    
    def __init__(
        self,
        vectorstore,
        documents: List[Document],
        alpha: float = 0.5
    ):
        """
        Initialize hybrid retriever
        
        Args:
            vectorstore: Vector database instance
            documents: All documents for BM25 indexing
            alpha: Weight for semantic vs keyword (0=keyword, 1=semantic)
        """
        self.vectorstore = vectorstore
        self.documents = documents
        self.alpha = alpha
        
        # Build BM25 index
        tokenized_docs = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Document]:
        """
        Perform hybrid retrieval
        
        Args:
            query: Search query
            k: Number of results
            filter_dict: Optional metadata filters
            
        Returns:
            Top-k most relevant documents
        """
        # Semantic search
        semantic_results = self.vectorstore.similarity_search_with_score(
            query, k=k*2, filter=filter_dict
        )
        
        # BM25 keyword search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize scores
        semantic_docs = {i: (doc, score) for i, (doc, score) in enumerate(semantic_results)}
        
        # Create document index mapping
        doc_to_idx = {id(doc.page_content): i for i, doc in enumerate(self.documents)}
        
        # Combine scores
        combined_scores = {}
        
        # Add semantic scores
        for doc, sem_score in semantic_results:
            doc_idx = doc_to_idx.get(id(doc.page_content))
            if doc_idx is not None:
                # Normalize semantic score (assuming lower is better for distance)
                norm_sem = 1 / (1 + sem_score)
                combined_scores[doc_idx] = self.alpha * norm_sem
        
        # Add BM25 scores
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        for idx, bm25_score in enumerate(bm25_scores):
            norm_bm25 = bm25_score / max_bm25
            if idx in combined_scores:
                combined_scores[idx] += (1 - self.alpha) * norm_bm25
            else:
                combined_scores[idx] = (1 - self.alpha) * norm_bm25
        
        # Filter by metadata if needed
        if filter_dict:
            filtered_scores = {}
            for idx, score in combined_scores.items():
                doc = self.documents[idx]
                if all(doc.metadata.get(k) == v for k, v in filter_dict.items()):
                    filtered_scores[idx] = score
            combined_scores = filtered_scores
        
        # Get top-k
        top_indices = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x],
            reverse=True
        )[:k]
        
        return [self.documents[idx] for idx in top_indices]


class RAGRetrieval:
    """
    LangChain-based RAG retrieval with multiple query modes
    """
    
    def __init__(
        self,
        vectorstore,
        llm_provider: str = "groq",
        model_name: Optional[str] = None,
        temperature: float = 0.0
    ):
        """
        Initialize RAG retrieval system
        
        Args:
            vectorstore: Vector database instance
            llm_provider: "groq" (recommended, fast & free) or "google"
            model_name: Specific model name
            temperature: LLM temperature
        """
        self.vectorstore = vectorstore
        
        # Initialize LLM - Groq is faster and has free tier
        if llm_provider == "groq":
            model_name = model_name or "llama-3.3-70b-versatile"  # Fast and smart
            self.llm = ChatGroq(
                model=model_name,
                temperature=temperature,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
        elif llm_provider == "google":
            model_name = model_name or "gemini-pro"
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    
    def query_single_paper(
        self,
        question: str,
        paper_id: str,
        k: int = 5
    ) -> Dict:
        """
        Query a single paper
        
        Args:
            question: User question
            paper_id: Target paper ID
            k: Number of chunks to retrieve
            
        Returns:
            Answer dictionary with sources
        """
        # Pre-fetch scored retrieval diagnostics
        scored_results = self.vectorstore.similarity_search_with_score(
            question,
            k=k,
            filter={"paper_id": paper_id}
        )

        retrieval_diagnostics = []
        for doc, distance in scored_results:
            relevance = 1 / (1 + float(distance))
            retrieval_diagnostics.append({
                "paper_id": doc.metadata.get("paper_id", "Unknown"),
                "paper_title": doc.metadata.get("paper_title", "Unknown"),
                "section": doc.metadata.get("section", "Unknown"),
                "chunk_index": doc.metadata.get("chunk_index", "Unknown"),
                "distance": float(distance),
                "relevance_score": float(relevance),
                "preview": doc.page_content[:220].replace("\n", " ")
            })

        avg_relevance = (
            sum(item["relevance_score"] for item in retrieval_diagnostics) / len(retrieval_diagnostics)
            if retrieval_diagnostics else 0.0
        )

        # Create retriever with paper filter
        retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": k,
                "filter": {"paper_id": paper_id}
            }
        )
        
        # Create prompt template
        template = """Use the following pieces of context from a scientific paper to answer the question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer: Let me provide a detailed answer based on the paper's content:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        # Execute query
        result = qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"],
            "retrieval_diagnostics": retrieval_diagnostics,
            "retrieval_summary": {
                "retrieval_mode": "vector_similarity",
                "k": k,
                "avg_relevance": avg_relevance
            }
        }
    
    def query_multi_paper(
        self,
        question: str,
        k: int = 10
    ) -> Dict:
        """
        Query across multiple papers
        
        Args:
            question: User question
            k: Number of chunks to retrieve
            
        Returns:
            Synthesized answer with sources from multiple papers
        """
        # Pre-fetch scored retrieval diagnostics
        scored_results = self.vectorstore.similarity_search_with_score(
            question,
            k=k
        )

        retrieval_diagnostics = []
        for doc, distance in scored_results:
            relevance = 1 / (1 + float(distance))
            retrieval_diagnostics.append({
                "paper_id": doc.metadata.get("paper_id", "Unknown"),
                "paper_title": doc.metadata.get("paper_title", "Unknown"),
                "section": doc.metadata.get("section", "Unknown"),
                "chunk_index": doc.metadata.get("chunk_index", "Unknown"),
                "distance": float(distance),
                "relevance_score": float(relevance),
                "preview": doc.page_content[:220].replace("\n", " ")
            })

        avg_relevance = (
            sum(item["relevance_score"] for item in retrieval_diagnostics) / len(retrieval_diagnostics)
            if retrieval_diagnostics else 0.0
        )

        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
        
        # Create prompt template for synthesis
        template = """You are synthesizing information from multiple scientific papers.
Use the following pieces of context to answer the question comprehensively.
If different papers have different findings, mention the differences.

Context from multiple papers:
{context}

Question: {question}

Synthesized Answer (mention which papers support each point):"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        # Execute query
        result = qa_chain({"query": question})
        
        # Group sources by paper
        papers_cited = {}
        for doc in result["source_documents"]:
            paper_id = doc.metadata.get("paper_id", "Unknown")
            paper_title = doc.metadata.get("paper_title", "Unknown")
            if paper_id not in papers_cited:
                papers_cited[paper_id] = paper_title
        
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"],
            "papers_cited": papers_cited,
            "retrieval_diagnostics": retrieval_diagnostics,
            "retrieval_summary": {
                "retrieval_mode": "vector_similarity",
                "k": k,
                "avg_relevance": avg_relevance
            }
        }
    
    def hybrid_search_query(
        self,
        question: str,
        documents: List[Document],
        k: int = 5,
        alpha: float = 0.5
    ) -> Dict:
        """
        Query using hybrid retrieval
        
        Args:
            question: User question
            documents: All documents for BM25
            k: Number of results
            alpha: Semantic vs keyword weight
            
        Returns:
            Answer with hybrid-retrieved sources
        """
        # Create hybrid retriever
        hybrid_retriever = HybridRetriever(
            self.vectorstore,
            documents,
            alpha=alpha
        )
        
        # Retrieve documents
        retrieved_docs = hybrid_retriever.hybrid_search(question, k=k)
        
        # Create context from retrieved documents
        context = "\n\n".join([
            f"[Source {i+1}] {doc.page_content}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        # Create prompt
        prompt = f"""Answer the following question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""
        
        # Get LLM response
        response = self.llm.invoke(prompt).content
        
        return {
            "answer": response,
            "source_documents": retrieved_docs
        }


