"""
Vector Database Abstraction Layer
Supports multiple vector database backends with unified interface
"""

from typing import List, Dict, Optional, Union
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
import os


class VectorDBFactory:
    """
    Factory for creating different vector database backends
    """
    
    @staticmethod
    def create_vectorstore(
        db_type: str,
        embeddings,
        documents: Optional[List[Document]] = None,
        **kwargs
    ):
        """
        Create a vector store instance
        
        Args:
            db_type: Type of vector database ("chroma", "qdrant", "pinecone")
            embeddings: Embedding function
            documents: Optional documents to initialize with
            **kwargs: Database-specific configuration
            
        Returns:
            Vector store instance
        """
        if db_type == "chroma":
            return VectorDBFactory._create_chroma(embeddings, documents, **kwargs)
        elif db_type == "qdrant":
            return VectorDBFactory._create_qdrant(embeddings, documents, **kwargs)
        elif db_type == "pinecone":
            return VectorDBFactory._create_pinecone(embeddings, documents, **kwargs)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    @staticmethod
    def _create_chroma(embeddings, documents, **kwargs):
        """
        Create Chroma vector store
        
        Configuration example:
            persist_directory: "./chroma_db"
            collection_name: "papers"
        """
        persist_directory = kwargs.get('persist_directory', './chroma_db')
        collection_name = kwargs.get('collection_name', 'papers')
        
        if documents:
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=persist_directory,
                collection_name=collection_name
            )
        else:
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                collection_name=collection_name
            )
        
        print(f"✓ Chroma vector store initialized at {persist_directory}")
        return vectorstore
    
    @staticmethod
    def _create_qdrant(embeddings, documents, **kwargs):
        """
        Create Qdrant vector store
        
        Configuration example:
            url: "http://localhost:6333"
            collection_name: "papers"
            api_key: Optional API key for Qdrant Cloud
        
        Note: Requires qdrant-client package
        """
        try:
            from langchain_community.vectorstores import Qdrant
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError(
                "Qdrant requires: pip install qdrant-client"
            )
        
        url = kwargs.get('url', 'http://localhost:6333')
        collection_name = kwargs.get('collection_name', 'papers')
        api_key = kwargs.get('api_key', None)
        
        client = QdrantClient(
            url=url,
            api_key=api_key
        )
        
        if documents:
            vectorstore = Qdrant.from_documents(
                documents=documents,
                embedding=embeddings,
                url=url,
                collection_name=collection_name,
                api_key=api_key
            )
        else:
            vectorstore = Qdrant(
                client=client,
                collection_name=collection_name,
                embeddings=embeddings
            )
        
        print(f"✓ Qdrant vector store initialized at {url}")
        return vectorstore
    
    @staticmethod
    def _create_pinecone(embeddings, documents, **kwargs):
        """
        Create Pinecone vector store
        
        Configuration example:
            index_name: "papers"
            environment: "us-west1-gcp"
            api_key: Pinecone API key
        
        Note: Requires pinecone-client package
        """
        try:
            from langchain_community.vectorstores import Pinecone
            import pinecone
        except ImportError:
            raise ImportError(
                "Pinecone requires: pip install pinecone-client"
            )
        
        api_key = kwargs.get('api_key', os.getenv('PINECONE_API_KEY'))
        environment = kwargs.get('environment', os.getenv('PINECONE_ENV'))
        index_name = kwargs.get('index_name', 'papers')
        
        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)
        
        if documents:
            vectorstore = Pinecone.from_documents(
                documents=documents,
                embedding=embeddings,
                index_name=index_name
            )
        else:
            vectorstore = Pinecone.from_existing_index(
                index_name=index_name,
                embedding=embeddings
            )
        
        print(f"✓ Pinecone vector store initialized: {index_name}")
        return vectorstore


class UnifiedVectorDB:
    """
    Unified interface for different vector databases
    """
    
    def __init__(
        self,
        db_type: str = "chroma",
        embeddings = None,
        **config
    ):
        """
        Initialize unified vector database
        
        Args:
            db_type: "chroma", "qdrant", or "pinecone"
            embeddings: Embedding function
            **config: Database-specific configuration
        """
        self.db_type = db_type
        self.embeddings = embeddings
        self.config = config
        self.vectorstore = None
    
    def create(self, documents: List[Document]):
        """
        Create new vector store with documents
        
        Args:
            documents: List of Document objects
        """
        self.vectorstore = VectorDBFactory.create_vectorstore(
            db_type=self.db_type,
            embeddings=self.embeddings,
            documents=documents,
            **self.config
        )
        return self.vectorstore
    
    def load(self):
        """
        Load existing vector store
        """
        self.vectorstore = VectorDBFactory.create_vectorstore(
            db_type=self.db_type,
            embeddings=self.embeddings,
            documents=None,
            **self.config
        )
        return self.vectorstore
    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to existing store
        
        Args:
            documents: List of Document objects
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        self.vectorstore.add_documents(documents)
        print(f"✓ Added {len(documents)} documents")
    
    def search(
        self,
        query: str,
        k: int = 5,
        **kwargs
    ) -> List[Document]:
        """
        Similarity search
        
        Args:
            query: Search query
            k: Number of results
            **kwargs: Additional search parameters
            
        Returns:
            List of relevant documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        return self.vectorstore.similarity_search(query, k=k, **kwargs)
    
    def search_with_score(
        self,
        query: str,
        k: int = 5,
        **kwargs
    ) -> List[tuple]:
        """
        Similarity search with scores
        
        Args:
            query: Search query
            k: Number of results
            **kwargs: Additional search parameters
            
        Returns:
            List of (document, score) tuples
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        return self.vectorstore.similarity_search_with_score(query, k=k, **kwargs)
    
    def as_retriever(self, **kwargs):
        """
        Get retriever interface
        
        Args:
            **kwargs: Retriever configuration
            
        Returns:
            Retriever object
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        return self.vectorstore.as_retriever(**kwargs)


def get_default_config(db_type: str) -> Dict:
    """
    Get default configuration for each database type
    
    Args:
        db_type: Database type
        
    Returns:
        Default configuration dictionary
    """
    configs = {
        "chroma": {
            "persist_directory": "./chroma_db",
            "collection_name": "papers"
        },
        "qdrant": {
            "url": "http://localhost:6333",
            "collection_name": "papers",
            "api_key": None  # For Qdrant Cloud
        },
        "pinecone": {
            "index_name": "papers",
            "api_key": os.getenv("PINECONE_API_KEY"),
            "environment": os.getenv("PINECONE_ENV", "us-west1-gcp")
        }
    }
    
    return configs.get(db_type, {})


def main():
    """Example usage"""
    from embeddings import EmbeddingManager
    import json
    
    # Load chunks
    with open('data/chunks.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Initialize embeddings
    embedding_manager = EmbeddingManager(provider="openai")
    documents = embedding_manager.chunks_to_documents(chunks)
    
    # Example 1: Using Chroma (default)
    print("\n=== Using Chroma ===")
    chroma_db = UnifiedVectorDB(
        db_type="chroma",
        embeddings=embedding_manager.embeddings,
        persist_directory="./chroma_db"
    )
    chroma_db.create(documents)
    
    # Example 2: Switch to Qdrant (commented - requires Qdrant server)
    # print("\n=== Using Qdrant ===")
    # qdrant_db = UnifiedVectorDB(
    #     db_type="qdrant",
    #     embeddings=embedding_manager.embeddings,
    #     url="http://localhost:6333"
    # )
    # qdrant_db.create(documents)
    
    # Example 3: Switch to Pinecone (commented - requires API key)
    # print("\n=== Using Pinecone ===")
    # pinecone_db = UnifiedVectorDB(
    #     db_type="pinecone",
    #     embeddings=embedding_manager.embeddings,
    #     index_name="papers"
    # )
    # pinecone_db.create(documents)


if __name__ == "__main__":
    main()
