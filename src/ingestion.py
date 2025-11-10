"""
PDF Ingestion Module for Scientific Papers
Handles PDF parsing, metadata extraction, and section-based text segmentation
"""

import fitz  # PyMuPDF
import re
from typing import Dict, List, Optional
from pathlib import Path
import json


class PaperIngestion:
    """
    Extracts structured information from scientific papers in PDF format
    """
    
    # Common section headers in scientific papers
    SECTION_PATTERNS = [
        r'^abstract\s*$',
        r'^introduction\s*$',
        r'^methods?\s*$',
        r'^methodology\s*$',
        r'^results?\s*$',
        r'^discussion\s*$',
        r'^conclusion\s*$',
        r'^references?\s*$',
        r'^bibliography\s*$',
        r'^literature cited\s*$',
        r'^related work\s*$',
        r'^background\s*$',
        r'^experiments?\s*$',
        r'^evaluation\s*$',
        r'^future work\s*$',
        r'^limitations?\s*$'
    ]
    
    def __init__(self):
        self.section_regex = re.compile(
            '|'.join(self.SECTION_PATTERNS),
            re.IGNORECASE | re.MULTILINE
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract raw text from PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        doc = fitz.open(pdf_path)
        text = ""
        
        for page in doc:
            text += page.get_text()
        
        doc.close()
        return text
    
    def extract_metadata(self, pdf_path: str, text: str) -> Dict[str, str]:
        """
        Extract metadata from PDF and text heuristics
        
        Args:
            pdf_path: Path to the PDF file
            text: Extracted text content
            
        Returns:
            Dictionary containing metadata
        """
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        doc.close()
        
        # Extract from PDF metadata
        title = metadata.get('title', '')
        author = metadata.get('author', '')
        
        # Fallback: extract from first page heuristics
        first_page_lines = text.split('\n')[:50]
        
        if not title:
            # Title is usually in first few lines, often in larger font
            # Simple heuristic: first non-empty line with more than 10 chars
            for line in first_page_lines:
                line = line.strip()
                if len(line) > 10 and not line.startswith('http'):
                    title = line
                    break
        
        if not author:
            # Look for common author patterns
            for line in first_page_lines:
                if re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', line):
                    author = line.strip()
                    break
        
        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', ' '.join(first_page_lines))
        year = year_match.group(0) if year_match else 'Unknown'
        
        return {
            'title': title or Path(pdf_path).stem,
            'author': author or 'Unknown',
            'year': year,
            'filename': Path(pdf_path).name
        }
    
    def split_into_sections(self, text: str) -> Dict[str, str]:
        """
        Split paper text into sections based on common headers
        
        Args:
            text: Full paper text
            
        Returns:
            Dictionary mapping section names to their content
        """
        sections = {}
        lines = text.split('\n')
        
        current_section = 'introduction'
        current_content = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if this line is a section header
            if self.section_regex.match(line_stripped) and len(line_stripped) < 50:
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line_stripped.lower()
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def process_paper(self, pdf_path: str, paper_id: Optional[str] = None) -> Dict:
        """
        Complete pipeline: extract text, metadata, and sections
        
        Args:
            pdf_path: Path to the PDF file
            paper_id: Optional unique identifier for the paper
            
        Returns:
            Dictionary containing all extracted information
        """
        if paper_id is None:
            paper_id = Path(pdf_path).stem
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        # Extract metadata
        metadata = self.extract_metadata(pdf_path, text)
        
        # Split into sections
        sections = self.split_into_sections(text)
        
        return {
            'paper_id': paper_id,
            'metadata': metadata,
            'sections': sections,
            'full_text': text
        }
    
    def process_multiple_papers(self, pdf_paths: List[str]) -> List[Dict]:
        """
        Process multiple papers
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            List of processed paper dictionaries
        """
        papers = []
        
        for pdf_path in pdf_paths:
            try:
                paper_data = self.process_paper(pdf_path)
                papers.append(paper_data)
                print(f"✓ Processed: {paper_data['metadata']['title'][:60]}...")
            except Exception as e:
                print(f"✗ Error processing {pdf_path}: {str(e)}")
        
        return papers
    
    def save_to_json(self, papers: List[Dict], output_path: str):
        """
        Save processed papers to JSON file
        
        Args:
            papers: List of processed paper dictionaries
            output_path: Path to output JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(papers)} papers to {output_path}")


def main():
    """Example usage"""
    ingestion = PaperIngestion()
    
    # Process a single paper
    paper = ingestion.process_paper("data/sample_paper.pdf")
    print(f"\nPaper: {paper['metadata']['title']}")
    print(f"Sections found: {list(paper['sections'].keys())}")
    
    # Save to JSON
    ingestion.save_to_json([paper], "data/processed_papers.json")


if __name__ == "__main__":
    main()
