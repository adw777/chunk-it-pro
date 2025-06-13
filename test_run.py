#### test chunking pipeline ####
import asyncio
from chunk_it_pro import SemanticChunkingPipeline

async def main():
    # Initialize the pipeline
    pipeline = SemanticChunkingPipeline()
    try:
        # Process your document
        initial_chunks, semantic_chunks, threshold = await pipeline.process_document(
            file_path="legal_order.pdf",  # or .txt, .docx, .md
                embedding_provider="voyage",    # or "openai"
                verbose=True
            )
        print(f"Created {len(semantic_chunks)} semantic chunks")
        for i, chunk in enumerate(semantic_chunks[:5]):  # Show first 5
            print(f"\nChunk {i+1}: {chunk[:200]}...")
        print(f"Threshold: {threshold}")

    except Exception as e:
        print(f"Error: {e}")
        return

asyncio.run(main())

#### test quick_chunking ####
import asyncio
from chunk_it_pro.pipeline import chunk_document

async def quick_chunk():
    # One-line document processing
    initial, semantic, threshold = await chunk_document("legal_order.pdf")
    return semantic

# Get chunks quickly
chunks = asyncio.run(quick_chunk())             

#### test process_directory ####
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
        output_file = f"semantic_chunks_{pdf_file.stem}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(semantic):
                f.write(f"SEMANTIC_CHUNK_{i+1}:\n{chunk}\n\n")

asyncio.run(process_directory())