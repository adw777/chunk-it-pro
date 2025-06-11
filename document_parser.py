import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Union
import fitz  # PyMuPDF
from docx import Document
import mammoth


class DocumentParser:
    """Parse various document formats and convert to markdown"""
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.txt', '.docx', '.md'}
    
    def parse_document(self, file_path: Union[str, Path]) -> str:
        """Parse document and return markdown content"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported format: {file_path.suffix}")
        
        if file_path.suffix.lower() == '.pdf':
            return self._parse_pdf(file_path)
        elif file_path.suffix.lower() == '.docx':
            return self._parse_docx(file_path)
        elif file_path.suffix.lower() == '.txt':
            return self._parse_txt(file_path)
        elif file_path.suffix.lower() == '.md':
            return self._parse_markdown(file_path)
    
    def _parse_pdf(self, file_path: Path) -> str:
        """Parse PDF and extract text with structure"""
        doc = fitz.open(file_path)
        markdown_content = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Add page break marker
            if page_num > 0:
                markdown_content.append(f"\n\n---\n**Page {page_num + 1}**\n\n")
            
            # Process text blocks
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            text_content = span["text"].strip()
                            if text_content:
                                # Check for headers based on font size
                                if span["size"] > 14:
                                    line_text += f"# {text_content}\n"
                                elif span["size"] > 12:
                                    line_text += f"## {text_content}\n"
                                else:
                                    line_text += f"{text_content} "
                        
                        if line_text.strip():
                            markdown_content.append(line_text.strip())
        
        doc.close()
        return "\n".join(markdown_content)
    
    def _parse_docx(self, file_path: Path) -> str:
        """Parse DOCX and convert to markdown"""
        with open(file_path, "rb") as docx_file:
            result = mammoth.convert_to_markdown(docx_file)
            return result.value
    
    def _parse_txt(self, file_path: Path) -> str:
        """Parse text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add basic markdown structure for plain text
        lines = content.split('\n')
        markdown_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Simple heuristic for headers
                if line.isupper() and len(line) < 100:
                    markdown_lines.append(f"# {line}")
                elif line.endswith(':') and len(line) < 100:
                    markdown_lines.append(f"## {line}")
                else:
                    markdown_lines.append(line)
            else:
                markdown_lines.append("")
        
        return "\n".join(markdown_lines)
    
    def _parse_markdown(self, file_path: Path) -> str:
        """Parse markdown file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def identify_breakpoints(self, markdown_text: str) -> List[Tuple[int, str]]:
        """Identify structural breakpoints in markdown text"""
        lines = markdown_text.split('\n')
        breakpoints = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Headers
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                breakpoints.append((i, f"header_h{level}"))
            
            # Page breaks
            elif line.startswith('---') or 'page' in line.lower():
                breakpoints.append((i, "page_break"))
            
            # Horizontal rules
            elif line.startswith('---') or line.startswith('***'):
                breakpoints.append((i, "section_break"))
        
        return breakpoints