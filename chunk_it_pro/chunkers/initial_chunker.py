from typing import List, Tuple
import tiktoken

from ..config import Config


class InitialChunker:
    """Create initial chunks of specified token size"""
    
    def __init__(self, tokenizer_name: str = None, chunk_size: int = None):
        self.tokenizer = tiktoken.get_encoding(tokenizer_name or Config.TOKENIZER_NAME)
        self.chunk_size = chunk_size or Config.INITIAL_CHUNK_SIZE
    
    def create_initial_chunks(self, markdown_text: str, breakpoints: List[Tuple[int, str]]) -> List[str]:
        """Create initial chunks respecting breakpoints"""
        lines = markdown_text.split('\n')
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        # Sort breakpoints by line number
        breakpoints_dict = {line_num: break_type for line_num, break_type in breakpoints}
        
        for i, line in enumerate(lines):
            line_tokens = len(self.tokenizer.encode(line))
            
            # Check if adding this line would exceed chunk size
            if current_tokens + line_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            # Check for breakpoints
            if i in breakpoints_dict and current_chunk:
                # Major breakpoint - end current chunk
                if breakpoints_dict[i] in ['header_h1', 'header_h2', 'page_break']:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return self._filter_valid_chunks(chunks)
    
    def _filter_valid_chunks(self, chunks: List[str]) -> List[str]:
        """Filter out very small or empty chunks"""
        valid_chunks = []
        
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk and len(self.tokenizer.encode(chunk)) >= Config.MIN_CHUNK_SIZE:
                valid_chunks.append(chunk)
        
        return valid_chunks
    
    def save_chunks(self, chunks: List[str], output_file: str = None) -> List[str]:
        """Save chunks to file"""
        if output_file is None:
            output_file = Config.DEFAULT_OUTPUT_FILES["initial_chunks"]
            
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                f.write(f"CHUNK_{i+1}:\n{chunk}\n\n{'='*50}\n\n")
        
        print(f"Saved {len(chunks)} initial chunks to {output_file}")
        return chunks 