import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Union

from .omniparse import parse_file_with_omniparse
from ..config import Config


class DocumentParser:
    """Parse various document formats and convert to markdown using omniparse"""
    
    def __init__(self, api_url: str = None):
        self.supported_formats = Config.SUPPORTED_FORMATS
        self.api_url = api_url or Config.OMNIPARSE_API_URL
    
    def parse_document(self, file_path: Union[str, Path]) -> str:
        """Parse document and return markdown content using omniparse"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported format: {file_path.suffix}")
        
        try:
            content = parse_file_with_omniparse(str(file_path), self.api_url)
            return self._ensure_markdown_format(content, file_path.suffix.lower())
        except Exception as e:
            raise ValueError(f"Failed to parse document {file_path}: {str(e)}")
    
    def _ensure_markdown_format(self, content: str, file_extension: str) -> str:
        """Ensure content is in markdown format"""
        if file_extension == '.md':
            return content
        elif file_extension == '.txt':
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
        else:
            # For PDF and DOCX, assume the API returns properly formatted content
            return content
    
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