"""
Embeddings Module for Text Vectorization
Handles embedding generation and vector store integration
"""

from typing import List, Dict, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os


class EmbeddingManager:
    """
    Manages embeddings and vector store operations
    """
    
    def __init__(
        self,
        provider: str = "huggingface",
        model: Optional[str] = None,
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize embedding manager
        
        Args:
            provider: "huggingface" (free, local) or "google" (requires API key)
            model: Specific model name (optional)
            persist_directory: Path to persist Chroma database
        """
        self.provider = provider
        self.persist_directory = persist_directory
        
        # Initialize embeddings based on provider
        if provider == "google":
            model = model or "models/embedding-001"
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=model,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        elif provider == "huggingface":
            # Use all-MiniLM-L6-v2: fast, small, good performance
            model = model or "sentence-transformers/all-MiniLM-L6-v2"
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self.vectorstore = None
    
    def chunks_to_documents(self, chunks: List[Dict]) -> List[Document]:
        """
        Convert chunk dictionaries to LangChain Document objects
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of Document objects
        """
        documents = []
        
        for chunk in chunks:
            # Extract text
            text = chunk.get('text', '')
            
            # Create metadata (exclude text from metadata)
            metadata = {k: v for k, v in chunk.items() if k != 'text'}
            
            # Create Document
            doc = Document(
                page_content=text,
                metadata=metadata
            )
            documents.append(doc)
        
        return documents
    
    def create_vectorstore(
        self,
        chunks: List[Dict],
        collection_name: str = "papers"
    ) -> Chroma:
        """
        Create a new Chroma vector store from chunks
        
        Args:
            chunks: List of chunk dictionaries
            collection_name: Name for the collection
            
        Returns:
            Chroma vector store instance
        """
        # Convert chunks to documents
        documents = self.chunks_to_documents(chunks)
        
        print(f"Creating vector store with {len(documents)} documents...")
        
        # Create Chroma vector store
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=collection_name
        )
        
        print(f"✓ Vector store created and persisted to {self.persist_directory}")
        
        return self.vectorstore
    
    def load_vectorstore(self, collection_name: str = "papers") -> Chroma:
        """
        Load existing Chroma vector store
        
        Args:
            collection_name: Name of the collection to load
            
        Returns:
            Chroma vector store instance
        """
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        
        print(f"✓ Loaded vector store from {self.persist_directory}")
        
        return self.vectorstore
    
    def add_chunks(self, chunks: List[Dict]) -> None:
        """
        Add new chunks to existing vector store
        
        Args:
            chunks: List of chunk dictionaries
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        
        documents = self.chunks_to_documents(chunks)
        self.vectorstore.add_documents(documents)
        
        print(f"✓ Added {len(documents)} documents to vector store")
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Document]:
        """
        Perform similarity search
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of most similar documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")
        
        if filter_dict:
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter=filter_dict
            )
        else:
            results = self.vectorstore.similarity_search(query, k=k)
        
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[tuple]:
        """
        Perform similarity search with relevance scores
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of (document, score) tuples
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")
        
        if filter_dict:
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=k,
                filter=filter_dict
            )
        else:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        return results
    
    def get_retriever(self, search_kwargs: Optional[Dict] = None):
        """
        Get a LangChain retriever interface
        
        Args:
            search_kwargs: Arguments for retrieval (k, filter, etc.)
            
        Returns:
            Retriever object
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")
        
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def delete_collection(self, collection_name: str = "papers"):
        """
        Delete a collection from the vector store
        
        Args:
            collection_name: Name of collection to delete
        """
        if self.vectorstore is not None:
            self.vectorstore.delete_collection()
            self.vectorstore = None
            print(f"✓ Deleted collection: {collection_name}")


