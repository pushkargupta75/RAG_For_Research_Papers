"""
Text Chunking Module with Token-Aware Splitting
Handles intelligent text chunking with overlap for RAG pipelines
"""

import tiktoken
from typing import List, Dict, Optional
import re


class TextChunker:
    """
    Token-aware text chunking with configurable overlap
    """
    
    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        chunk_size: int = 400,
        chunk_overlap: int = 100
    ):
        """
        Initialize chunker with tokenizer
        
        Args:
            encoding_name: tiktoken encoding (cl100k_base for GPT-3.5/4)
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
        """
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Split text into overlapping chunks based on token count
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries
        """
        # Encode text into tokens
        tokens = self.encoding.encode(text)
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            # Extract chunk
            end_idx = start_idx + self.chunk_size
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Create chunk metadata
            chunk_data = {
                'text': chunk_text,
                'token_count': len(chunk_tokens),
                'start_token': start_idx,
                'end_token': end_idx
            }
            
            # Add custom metadata
            if metadata:
                chunk_data.update(metadata)
            
            chunks.append(chunk_data)
            
            # Move to next chunk with overlap
            start_idx += (self.chunk_size - self.chunk_overlap)
        
        return chunks
    
    def chunk_by_sentences(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Chunk text by sentences, respecting token limits
        Creates more natural chunks at sentence boundaries
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If single sentence exceeds chunk_size, force split
            if sentence_tokens > self.chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk)
                    chunk_data = {
                        'text': chunk_text,
                        'token_count': current_tokens
                    }
                    if metadata:
                        chunk_data.update(metadata)
                    chunks.append(chunk_data)
                    current_chunk = []
                    current_tokens = 0
                
                # Split long sentence
                long_chunks = self.chunk_text(sentence, metadata)
                chunks.extend(long_chunks)
                continue
            
            # Check if adding sentence exceeds limit
            if current_tokens + sentence_tokens > self.chunk_size:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunk_data = {
                    'text': chunk_text,
                    'token_count': current_tokens
                }
                if metadata:
                    chunk_data.update(metadata)
                chunks.append(chunk_data)
                
                # Start new chunk with overlap
                overlap_tokens = 0
                overlap_sentences = []
                
                for i in range(len(current_chunk) - 1, -1, -1):
                    sent_tokens = self.count_tokens(current_chunk[i])
                    if overlap_tokens + sent_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, current_chunk[i])
                        overlap_tokens += sent_tokens
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_tokens = overlap_tokens
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Save last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_data = {
                'text': chunk_text,
                'token_count': current_tokens
            }
            if metadata:
                chunk_data.update(metadata)
            chunks.append(chunk_data)
        
        return chunks
    
    def chunk_paper(self, paper_data: Dict) -> List[Dict]:
        """
        Chunk all sections of a paper with metadata
        
        Args:
            paper_data: Paper dictionary from ingestion module
            
        Returns:
            List of chunks with paper and section metadata
        """
        all_chunks = []
        paper_id = paper_data['paper_id']
        metadata = paper_data['metadata']
        
        for section_name, section_text in paper_data['sections'].items():
            if not section_text.strip():
                continue
            
            chunk_metadata = {
                'paper_id': paper_id,
                'section': section_name,
                'paper_title': metadata['title'],
                'paper_author': metadata['author'],
                'paper_year': metadata['year']
            }
            
            # Chunk section text
            section_chunks = self.chunk_by_sentences(
                section_text,
                metadata=chunk_metadata
            )
            
            # Add chunk index within section
            for i, chunk in enumerate(section_chunks):
                chunk['chunk_index'] = i
            
            all_chunks.extend(section_chunks)
        
        return all_chunks
    
    def chunk_multiple_papers(self, papers_data: List[Dict]) -> List[Dict]:
        """
        Chunk multiple papers
        
        Args:
            papers_data: List of paper dictionaries
            
        Returns:
            Combined list of all chunks
        """
        all_chunks = []
        
        for paper_data in papers_data:
            paper_chunks = self.chunk_paper(paper_data)
            all_chunks.extend(paper_chunks)
            print(f"✓ Chunked: {paper_data['metadata']['title'][:60]}... "
                  f"({len(paper_chunks)} chunks)")
        
        return all_chunks
    
    def get_statistics(self, chunks: List[Dict]) -> Dict:
        """
        Get statistics about chunks
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Statistics dictionary
        """
        if not chunks:
            return {}
        
        token_counts = [c['token_count'] for c in chunks]
        
        return {
            'total_chunks': len(chunks),
            'total_tokens': sum(token_counts),
            'avg_tokens_per_chunk': sum(token_counts) / len(chunks),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'unique_papers': len(set(c.get('paper_id', '') for c in chunks)),
            'unique_sections': len(set(c.get('section', '') for c in chunks))
        }


