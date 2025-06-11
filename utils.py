"""Utility functions for semantic chunking pipeline"""

import os
import time
from functools import wraps
from typing import List, Dict, Any
import tiktoken


def timer(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"   {func.__name__} completed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper


def validate_file_path(file_path: str, supported_formats: set) -> bool:
    """Validate file path and format"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in supported_formats:
        raise ValueError(f"Unsupported format: {file_ext}. Supported: {supported_formats}")
    
    return True


def validate_openai_key(api_key: str = None) -> str:
    """Validate OpenAI API key"""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
    return key


def count_tokens(text: str, tokenizer_name: str = "cl100k_base") -> int:
    """Count tokens in text"""
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    return len(tokenizer.encode(text))


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def chunk_text_stats(chunks: List[str], tokenizer_name: str = "cl100k_base") -> Dict[str, Any]:
    """Calculate statistics for text chunks"""
    if not chunks:
        return {}
    
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    
    token_counts = [len(tokenizer.encode(chunk)) for chunk in chunks]
    word_counts = [len(chunk.split()) for chunk in chunks]
    char_counts = [len(chunk) for chunk in chunks]
    
    return {
        "count": len(chunks),
        "tokens": {
            "total": sum(token_counts),
            "average": sum(token_counts) / len(token_counts),
            "min": min(token_counts),
            "max": max(token_counts)
        },
        "words": {
            "total": sum(word_counts),
            "average": sum(word_counts) / len(word_counts),
            "min": min(word_counts),
            "max": max(word_counts)
        },
        "characters": {
            "total": sum(char_counts),
            "average": sum(char_counts) / len(char_counts),
            "min": min(char_counts),
            "max": max(char_counts)
        }
    }


def print_chunk_sample(chunks: List[str], num_samples: int = 3):
    """Print sample chunks for inspection"""
    if not chunks:
        print("No chunks to display")
        return
    
    print(f"\nSample chunks (showing {min(num_samples, len(chunks))} of {len(chunks)}):")
    print("-" * 50)
    
    for i in range(min(num_samples, len(chunks))):
        chunk = chunks[i]
        preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
        print(f"\nChunk {i+1}:")
        print(f"Length: {len(chunk)} chars")
        print(f"Preview: {preview}")
        print("-" * 30)


def estimate_openai_cost(num_tokens: int, model: str = "text-embedding-3-small") -> float:
    """Estimate OpenAI API cost"""
    # Pricing as of 2024 (per 1M tokens)
    pricing = {
        "text-embedding-3-small": 0.00002,
        "text-embedding-3-large": 0.00013,
        "text-embedding-ada-002": 0.00010
    }
    
    rate = pricing.get(model, 0.00002)  # Default to small model
    return (num_tokens / 1_000_000) * rate


def create_output_directory(base_dir: str = "output") -> str:
    """Create output directory with timestamp"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"semantic_chunking_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_pipeline_metadata(output_dir: str, metadata: Dict[str, Any]):
    """Save pipeline execution metadata"""
    import json
    
    metadata_file = os.path.join(output_dir, "pipeline_metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"Pipeline metadata saved to {metadata_file}")


def load_pipeline_metadata(metadata_file: str) -> Dict[str, Any]:
    """Load pipeline execution metadata"""
    import json
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        return json.load(f)