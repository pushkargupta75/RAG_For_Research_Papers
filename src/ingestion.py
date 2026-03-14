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

    def assess_academic_document(self, paper_data: Dict) -> Dict:
        """
        Assess whether a processed document looks like an academic,
        research-oriented document (e.g., research paper, review article,
        scholarly article, short communication).

        Heuristic-based scoring using section coverage, citation patterns,
        DOI/journal signals, and minimum content length.

        Args:
            paper_data: Processed paper dictionary from process_paper()

        Returns:
            Dictionary with validation result and diagnostic details.
            Key field `is_academic_document` indicates pass/fail.
        """
        full_text = paper_data.get("full_text", "") or ""
        sections = paper_data.get("sections", {}) or {}
        metadata = paper_data.get("metadata", {}) or {}

        section_names = [s.lower().strip() for s in sections.keys()]
        full_text_lower = full_text.lower()

        score = 0
        reasons = []
        signals = {}

        # 1) Minimum content length
        word_count = len(re.findall(r"\w+", full_text))
        signals["word_count"] = word_count
        if word_count >= 1200:
            score += 2
        elif word_count >= 600:
            score += 1
        else:
            reasons.append("Document is too short for a typical academic/research article.")

        # 2) Core scholarly section groups (more flexible than strict IMRaD)
        required_section_groups = {
            "abstract": ["abstract"],
            "intro": ["introduction", "background", "related work", "overview"],
            "methods": ["method", "methodology", "materials and methods", "experiments"],
            "results": ["results", "evaluation", "discussion"],
            "references": ["references", "bibliography", "literature cited", "works cited"]
        }

        matched_groups = 0
        for group_name, keywords in required_section_groups.items():
            matched = any(any(keyword in sec for keyword in keywords) for sec in section_names)
            signals[f"has_{group_name}"] = matched
            if matched:
                matched_groups += 1

        if matched_groups >= 4:
            score += 4
        elif matched_groups >= 3:
            score += 2
        elif matched_groups >= 2:
            score += 1
        else:
            reasons.append("Missing core academic sections (e.g., abstract/introduction/references).")

        # 3) Academic title / article type signals
        title = (metadata.get("title") or "").strip().lower()
        title_word_count = len(title.split()) if title else 0
        signals["title_word_count"] = title_word_count

        if title_word_count >= 4:
            score += 1

        title_keywords = [
            "study", "analysis", "framework", "approach", "method",
            "review", "survey", "evaluation", "comparison", "experiment"
        ]
        has_title_keyword = any(kw in title for kw in title_keywords)
        signals["title_academic_keyword"] = has_title_keyword
        if has_title_keyword:
            score += 1

        # 4) Citation-like patterns and publication signals
        citation_patterns = [
            r"\[[0-9]{1,3}\]",                 # [1], [23]
            r"\([A-Z][A-Za-z\-]+,\s*(19|20)\d{2}\)",  # (Smith, 2021)
            r"et\s+al\.",
            r"doi\s*:\s*10\.\d{4,9}/[-._;()/:A-Z0-9]+"
        ]

        citation_hits = 0
        for pattern in citation_patterns:
            if re.search(pattern, full_text, flags=re.IGNORECASE):
                citation_hits += 1

        signals["citation_signal_hits"] = citation_hits
        if citation_hits >= 2:
            score += 2
        elif citation_hits == 1:
            score += 1
        else:
            reasons.append("No strong citation/reference patterns were detected.")

        # 5) Additional scholarly text signals in body
        scholarly_terms = [
            "methodology", "dataset", "statistical", "hypothesis",
            "limitation", "future work", "peer review", "journal"
        ]
        scholarly_term_hits = sum(1 for term in scholarly_terms if term in full_text_lower)
        signals["scholarly_term_hits"] = scholarly_term_hits
        if scholarly_term_hits >= 3:
            score += 1

        # 6) Penalize obvious non-academic document types
        non_academic_terms = [
            "invoice", "purchase order", "resume", "curriculum vitae",
            "bank statement", "terms and conditions", "privacy policy",
            "user manual", "brochure", "meeting agenda"
        ]
        non_academic_hits = sum(1 for term in non_academic_terms if term in full_text_lower)
        signals["non_academic_term_hits"] = non_academic_hits
        if non_academic_hits >= 2:
            score -= 3
            reasons.append("Document contains strong non-academic patterns.")
        elif non_academic_hits == 1:
            score -= 1

        # Ensure score bounds
        score = max(0, score)

        # Threshold tuned to accept research-like articles while rejecting generic docs
        is_academic_document = score >= 6

        if is_academic_document:
            reasons = ["Document appears to be an academic/research-oriented article."]

        return {
            "is_academic_document": is_academic_document,
            "is_research_paper": is_academic_document,  # Backward compatibility
            "score": score,
            "max_score": 11,
            "reasons": reasons,
            "signals": signals
        }

    def assess_research_paper(self, paper_data: Dict) -> Dict:
        """
        Backward-compatible wrapper.

        Kept for existing callers; delegates to assess_academic_document.
        """
        return self.assess_academic_document(paper_data)
    
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


