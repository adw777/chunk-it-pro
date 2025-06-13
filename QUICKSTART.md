# ChunkIt Pro Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (optional but recommended)
export OPENAI_API_KEY="your-openai-key"
export VOYAGEAI_API_KEY="your-voyage-key"
```

## Basic Usage

### 1. Simple Document Chunking

```python
import asyncio
from chunk_it_pro import SemanticChunkingPipeline

async def main():
    # Initialize the pipeline
    pipeline = SemanticChunkingPipeline()
    
    # Process your document
    initial_chunks, semantic_chunks, threshold = await pipeline.process_document(
        file_path="your_document.pdf",  # or .txt, .docx, .md
        embedding_provider="voyage",    # or "openai"
        verbose=True
    )
    
    # Use the results
    print(f"Created {len(semantic_chunks)} semantic chunks")
    for i, chunk in enumerate(semantic_chunks[:3]):  # Show first 3
        print(f"\nChunk {i+1}: {chunk[:200]}...")

asyncio.run(main())
```

### 2. One-Line Chunking

```python
import asyncio
from chunk_it_pro.pipeline import chunk_document

async def quick_chunk():
    # One-line document processing
    initial, semantic, threshold = await chunk_document("document.pdf")
    return semantic

# Get chunks quickly
chunks = asyncio.run(quick_chunk())
```

### 3. Batch Processing

```python
import asyncio
from pathlib import Path
from chunk_it_pro import SemanticChunkingPipeline

async def process_directory():
    pipeline = SemanticChunkingPipeline()
    
    # Process all PDFs in a directory
    for pdf_file in Path("documents/").glob("*.pdf"):
        print(f"Processing {pdf_file.name}...")
        
        initial, semantic, threshold = await pipeline.process_document(
            file_path=str(pdf_file),
            save_files=False,  # Don't clutter with output files
            verbose=False
        )
        
        # Save results with meaningful names
        output_file = f"chunks_{pdf_file.stem}.txt"
        with open(output_file, 'w') as f:
            for i, chunk in enumerate(semantic):
                f.write(f"CHUNK_{i+1}:\n{chunk}\n\n")

asyncio.run(process_directory())
```

## Configuration Options

### Embedding Providers

```python
# VoyageAI (recommended for legal documents)
await pipeline.process_document("doc.pdf", embedding_provider="voyage")

# OpenAI (high quality, more expensive)
await pipeline.process_document("doc.pdf", embedding_provider="openai")

# Local model (no API costs, requires setup)
await pipeline.process_document("doc.pdf", embedding_provider="axon")
```

### Similarity Thresholds

```python
# Use 95th percentile (default)
await pipeline.process_document("doc.pdf", threshold_method="percentile", percentile=95)

# Use gradient-based detection
await pipeline.process_document("doc.pdf", threshold_method="gradient")

# Use local maxima detection
await pipeline.process_document("doc.pdf", threshold_method="local_maxima")
```

### Chunk Size Control

```python
# Larger chunks (up to 2048 tokens)
await pipeline.process_document("doc.pdf", max_chunk_len=2048)

# Smaller chunks (up to 512 tokens)
await pipeline.process_document("doc.pdf", max_chunk_len=512)
```

## Output Files

By default, the pipeline creates:

- `initial_chunks.txt` - Fixed-size chunks (256 tokens each)
- `semantic_chunks.txt` - Final semantic chunks
- `embeddings.npy` - Vector embeddings
- `cosine_distances.png` - Similarity visualization

Disable with `save_files=False`.

## Troubleshooting

### No API Keys
```python
# Check what's available
from chunk_it_pro import Config
status = Config.validate_config()
print(status)  # Shows which API keys are set
```

### Import Errors
```bash
# Run the test script
python test_installation.py
```
## Next Steps

1. Check out `example.py` for more detailed examples
2. Check out `test_run.py` for quick usage (single doc & multiple-docs)
2. Read `README.md` for complete documentation
3. Explore the `chunk_it_pro/` package structure
4. Customize configuration in `chunk_it_pro/config.py`

## Support

- Test your installation: `python test_installation.py`
- Example usage: `python example.py`
- Quick usage @test_run.py (first, comment out the code you dont want to run!)
- Configuration help: Check `chunk_it_pro/config.py` 