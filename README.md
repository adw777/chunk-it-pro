# ChunkIt Pro - Semantic Document Chunking Library

Python library for intelligent document chunking using semantic analysis. ChunkIt Pro breaks down documents into meaningful segments based on content similarity rather than arbitrary size limits.

## Features

- **Multiple Document Formats**: Supports PDF, DOCX, TXT, and Markdown files
- **Semantic Analysis**: Uses advanced embedding models to understand content similarity
- **Multiple Embedding Providers**: OpenAI, VoyageAI, and Sentence Transformers support
- **Intelligent Chunking**: Two-pass algorithm for optimal chunk boundaries
- **Configurable Thresholds**: Multiple methods for similarity threshold computation
- **Visual Analysis**: Generates plots showing similarity patterns
- **Easy Integration**: Simple API for quick integration into existing projects

## Installation

1. Clone the repository:
```bash
git clone https://github.com/adw777/chunk_it_pro
cd chunk_it_pro
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (create a `.env` file):
```env
OPENAI_API_KEY=your_openai_api_key_here
VOYAGEAI_API_KEY=your_voyage_api_key_here
OMNIPARSE_API_URL=your_omniparse_api_url_here
```

## Quick Start

### Basic Usage

```python
import asyncio
from chunk_it_pro import SemanticChunkingPipeline

async def main():
    # Initialize pipeline
    pipeline = SemanticChunkingPipeline()
    
    # Process document
    initial_chunks, semantic_chunks, threshold = await pipeline.process_document(
        file_path="your_document.pdf",
        embedding_provider="voyage",  # or "openai", "axon"
        threshold_method="percentile",
        percentile=95,
        max_chunk_len=1024
    )
    
    print(f"Created {len(semantic_chunks)} semantic chunks")
    print(f"Similarity threshold: {threshold:.4f}")

# Run the example
asyncio.run(main())
```

### Convenience Function

```python
import asyncio
from chunk_it_pro.pipeline import chunk_document

async def main():
    # Quick chunking with default settings
    initial_chunks, semantic_chunks, threshold = await chunk_document(
        file_path="document.pdf",
        embedding_provider="voyage"
    )
    
    # Use the chunks in your application
    for i, chunk in enumerate(semantic_chunks):
        print(f"Chunk {i+1}: {chunk[:100]}...")

asyncio.run(main())
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings | Optional* |
| `VOYAGEAI_API_KEY` | VoyageAI API key for embeddings | Optional* |
| `OMNIPARSE_API_URL` | Omniparse API URL for document parsing | Optional |

*At least one embedding provider API key is required.

### Embedding Providers

1. **VoyageAI** (Recommended for legal documents):
   - Model: `voyage-law-2`
   - High quality embeddings for legal text
   - Good performance/cost ratio

2. **OpenAI**:
   - Model: `text-embedding-3-small`
   - High quality general-purpose embeddings
   - More expensive but very reliable

3. **Axon** (Local):
   - Model: Fine-tuned legal embedding model
   - Runs locally, no API costs
   - Requires additional setup

### Threshold Methods

1. **Percentile** (Default): Uses the Nth percentile of cosine distances
2. **Gradient**: Finds points with highest gradient change
3. **Local Maxima**: Uses local maxima in distance patterns

## API Reference

### SemanticChunkingPipeline

Main class for semantic chunking operations.

```python
class SemanticChunkingPipeline:
    def __init__(self, openai_api_key: str = None, voyage_api_key: str = None)
    
    async def process_document(
        self,
        file_path: str,
        threshold_method: str = "percentile",
        percentile: float = 95,
        max_chunk_len: int = 1024,
        embedding_provider: str = "voyage",
        save_files: bool = True,
        verbose: bool = True
    ) -> Tuple[list, list, float]
    
    def get_chunk_statistics(self) -> Dict[str, Any]
    def print_statistics(self)
```

### chunk_document Function

Convenience function for quick document processing.

```python
async def chunk_document(
    file_path: str,
    embedding_provider: str = "voyage",
    threshold_method: str = "percentile",
    percentile: float = 95,
    max_chunk_len: int = 1024,
    save_files: bool = True,
    verbose: bool = True
) -> Tuple[list, list, float]
```

## Output Files

When `save_files=True`, the pipeline creates:

- `initial_chunks.txt`: Fixed-size initial chunks (256 tokens each)
- `semantic_chunks.txt`: Final semantic chunks
- `embeddings.npy`: Numpy array of embeddings
- `cosine_distances.png`: Visualization of similarity patterns

## Project Structure

```
chunk_it_pro/
├── chunk_it_pro/             # Main package
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration management
│   ├── pipeline.py           # Main pipeline implementation
│   ├── parsers/              # Document parsing modules
│   │   ├── __init__.py
│   │   ├── document_parser.py
│   │   └── omniparse.py
│   ├── chunkers/             # Chunking algorithms
│   │   ├── __init__.py
│   │   ├── initial_chunker.py
│   │   └── semantic_chunker.py
│   ├── embeddings/           # Embedding generation
│   │   ├── __init__.py
│   │   ├── embedding_analyzer.py
│   │   └── voyage_client.py
│   └── utils/                # Utility modules
│       ├── __init__.py
│       └── singleton.py
├── example.py                # Usage examples
├── requirements.txt          # Dependencies
└── README.md                # Documentation
```

## Algorithm Overview

ChunkIt uses a sophisticated two-pass algorithm:

### Pass 1: Initial Chunking
1. Parse document to markdown format
2. Identify structural breakpoints (headers, page breaks)
3. Create fixed-size chunks (256 tokens) respecting breakpoints
4. Generate embeddings for each chunk
5. Compute cosine distances between consecutive chunks
6. Determine similarity threshold using statistical methods

### Pass 2: Semantic Refinement
1. Split text into sentences
2. Generate sentence-level embeddings
3. Group sentences based on similarity threshold
4. Merge similar adjacent chunks (respecting max length)
5. Output final semantic chunks

## Advanced Usage

### Custom Configuration

```python
from chunk_it_pro.config import Config

# Override default settings
Config.INITIAL_CHUNK_SIZE = 512
Config.MAX_CHUNK_LENGTH = 2048
Config.DEFAULT_PERCENTILE = 90

# Check configuration
status = Config.validate_config()
print(status)
```

### Processing Multiple Documents

```python
import asyncio
from pathlib import Path
from chunk_it_pro import SemanticChunkingPipeline

async def process_multiple_documents():
    pipeline = SemanticChunkingPipeline()
    
    documents = Path("documents/").glob("*.pdf")
    
    for doc_path in documents:
        print(f"Processing {doc_path.name}...")
        
        initial_chunks, semantic_chunks, threshold = await pipeline.process_document(
            file_path=str(doc_path),
            save_files=True,
            verbose=False
        )
        
        # Save results with document name
        output_dir = Path("output") / doc_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Custom processing for each document...

asyncio.run(process_multiple_documents())
```

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure environment variables are set correctly
2. **Document Parsing Errors**: Check if Omniparse API is accessible
3. **Memory Issues**: Reduce batch size or chunk size for large documents
4. **NLTK Data**: The library automatically downloads required NLTK data

### Performance Tips

1. Use VoyageAI for best performance/cost balance
2. Adjust `max_chunk_len` based on your use case
3. Set `save_files=False` for better performance when processing many documents
4. Use local embedding models (Axon) for privacy-sensitive applications

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 